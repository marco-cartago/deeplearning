import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaConfig:
    def __init__(
        self, 
        vocab_size=0,      # Size of the tokenizer vocabulary (needed for embedding + LM head)
        d_model=256,       # Input/Output dimension of the block (size of the embedding)
        d_state=16,        # Latent state dimension (N in the paper)
        expand_factor=2,   # Expansion factor (E in the paper)
        d_conv=4,          # Width of the 1D causal convolution
        dt_rank: int | str ="auto",     # Rank for the step size projection (R in the paper)
        n_layers=8,
        norm_eps=1e-5,  
    ):
        """Configuration class for different Mamba components.

        Attributes
        ----------
        vocab_size: int
            Size of the tokenizer's vocabulary.
        d_model: int
            Number of features (size of the embedding in language models).
            D in the paper.
        d_state: int
            Size of the state space for a single feature. N in the paper.
            Total state-space size is D*N.
        expand_factor: int
            Expansion factor of the embedding size. E in the paper.
            Real size of the sequence before SSM is D*E.
        d_conv: int
            Small contextual convolution before the transformation.
        dt_rank: int | str
            Rank of a low rank linear projection of a sequence element to the Δ parameter.
            R in the paper.
        n_layers: int
            Depth of the Neural Network.
        norm_eps: float
            Epsilon for LayerNorm in case of zero variance layers.
        """
        self.vocab_size = vocab_size

        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = int(expand_factor * d_model) # Dimension inside the block (D*E)
        self.d_conv = d_conv

        self.n_layers = n_layers
        self.norm_eps = norm_eps
        
        # dt_rank is typically ceil(d_model / 16)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            if not isinstance(dt_rank, int):
                raise ValueError("'dt_rank' must be either int or 'auto'.")
            self.dt_rank = dt_rank


class SelectiveSSM(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner = config.d_inner
        d_state = config.d_state
        dt_rank = config.dt_rank

        # ----------------------------------------------------------------
        # 1. Core State Space Parameters (A and D)
        # ----------------------------------------------------------------
        # A matrix: (d_inner, d_state). 
        # Initialized to highly specific values to remember long contexts.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # Log enforces A remains positive during training
        
        # D vector: (d_inner). Skip connection directly from input to output.
        self.D = nn.Parameter(torch.ones(d_inner))

        # ----------------------------------------------------------------
        # 2. Input-Dependent Projections (The "Selective" part)
        # ----------------------------------------------------------------
        # Delta (Δ) step size projection. Uses a low-rank bottleneck.
        self.dt_proj = nn.Sequential(
            nn.Linear(d_inner, dt_rank, bias=False),
            nn.Linear(dt_rank, d_inner, bias=True)
        )
        
        # B and C projections. They project from the input to the state dimension.
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x shape: (Batch, Sequence_Length, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape
        device = x.device # Inherit CPU/GPU directly from the input

        # Retrieve the A matrix and ensure it is negative (stable recurrent system)
        A = -torch.exp(self.A_log.float()) # Shape: (d_inner, d_state)

        # ----------------------------------------------------------------
        # Generate Data-Dependent Parameters for the entire sequence
        # ----------------------------------------------------------------
        # Delta (Δ): Step size controls how much to remember/forget
        dt = self.dt_proj(x)                  # (Batch, Seq, d_inner)
        dt = F.softplus(dt)                   # Ensure step size is strictly positive
        
        # Predict B and C matrices based on the input x
        # B_and_C = self.x_proj(x)              # (Batch, Seq, d_state * 2)
        # B, C = torch.split(B_and_C, self.config.d_state, dim=-1) # Both are (Batch, Seq, d_state)
        B = self.B_proj(x)
        C = self.C_proj(x)

        # ----------------------------------------------------------------
        # The Explicit Recurrent Scan (Hardware-Aware algorithms hide this)
        # ----------------------------------------------------------------
        # Initialize the hidden state (h) to zeros. 
        # Shape: (Batch, d_inner, d_state)
        h = torch.zeros(batch_size, d_inner, self.config.d_state, device=device)
        ys = []

        # Iterate through time explicitly
        for t in range(seq_len):
            # Extract current timestep's variables
            xt = x[:, t, :]   # (Batch, d_inner)
            dt_t = dt[:, t, :] # (Batch, d_inner)
            bt = B[:, t, :]   # (Batch, d_state)
            ct = C[:, t, :]   # (Batch, d_state)

            # Discretize using Zero-Order Hold (ZOH)
            # 1. dA = exp(Δ * A)
            # Broadcasting dt_t (Batch, d_inner, 1) * A (1, d_inner, d_state)
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)) # (Batch, d_inner, d_state)
            
            # 2. dB = Δ * B
            # Broadcasting dt_t (Batch, d_inner, 1) * bt (Batch, 1, d_state)
            dB = dt_t.unsqueeze(-1) * bt.unsqueeze(1) # (Batch, d_inner, d_state)

            # Update Hidden State: h_t = dA * h_{t-1} + dB * x_t
            # xt is broadcasted to (Batch, d_inner, 1)
            h = dA * h + dB * xt.unsqueeze(-1)

            # Compute Output: y_t = C * h_t
            # Element-wise multiply h with ct (broadcasted), then sum across state dimension
            y_t = (h * ct.unsqueeze(1)).sum(dim=-1) # (Batch, d_inner)
            ys.append(y_t)

        # Re-stack the sequence: (Batch, Seq, d_inner)
        y = torch.stack(ys, dim=1)
        
        # Add the skip connection and return
        return y + (x * self.D)
    

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_inner = config.d_inner
        d_conv = config.d_conv

        # 1. Input Projection: Expands the input and creates two parallel branches
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 2. Causal 1D Convolution
        # padding=d_conv - 1 ensures we only look at past tokens, never future tokens.
        # groups=d_inner makes it a depthwise convolution (one filter per channel).
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True
        )

        # 3. The Selective State Space Model
        self.ssm = SelectiveSSM(config)

        # 4. Output Projection: Compresses back to d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x shape: (Batch, Sequence_Length, d_model)
        """
        seq_len = x.shape[1]

        # Step 1: Project up and split into Main branch and Gate branch
        x_proj = self.in_proj(x)
        x_main, x_gate = x_proj.chunk(2, dim=-1) # Both are (Batch, Seq, d_inner)

        # Step 2: Causal Convolution on the Main branch
        # PyTorch Conv1d expects (Batch, Channels, Length), so we transpose
        x_main = x_main.transpose(1, 2)
        
        # Apply convolution and immediately slice off the padding at the end 
        # to ensure strict causality (current token doesn't see future padding).
        x_main = self.conv1d(x_main)[:, :, :seq_len]
        
        # Transpose back to (Batch, Sequence_Length, Channels)
        x_main = x_main.transpose(1, 2)

        # Step 3: Activation (SiLU / Swish)
        x_main = F.silu(x_main)

        # Step 4: Run the Explicit SSM
        x_main = self.ssm(x_main)

        # Step 5: The Gating Mechanism
        # Activate the gate branch and multiply it element-wise with the main branch
        x_gate = F.silu(x_gate)
        x_combined = x_main * x_gate

        # Step 6: Output projection back to d_model
        out = self.out_proj(x_combined)
        
        return out


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Parametro apprendibile per scalare la normalizzazione (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Calcolo della varianza lungo l'ultima dimensione (d_model)
        variance = x.pow(2.).mean(-1, keepdim=True)
        # Normalizzazione
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed