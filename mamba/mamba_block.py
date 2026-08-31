import json
import math
import string
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaConfig:
    def __init__(
        self, 
        vocab_size=len(string.printable[:-3]),      # Size of the tokenizer vocabulary (needed for embedding + LM head)
        d_model=128,       # Input/Output dimension of the block (size of the embedding)
        d_state=8,        # Latent state dimension (N in the paper)
        expand_factor=2.,   # Expansion factor (E in the paper)
        d_conv=4,          # Width of the 1D causal convolution
        dt_rank: int | str ="auto",     # Rank for the step size projection (R in the paper)
        n_layers=4,
        norm_eps=1e-4,  
        charset_file: str | None = None
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
            R in the paper. `dt_rank`='auto' is R = D/16 rounded up.
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

        self.charset_file = charset_file
        
        # dt_rank is typically ceil(d_model / 16)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            if not isinstance(dt_rank, int):
                raise ValueError("'dt_rank' must be either int or 'auto'.")
            self.dt_rank = dt_rank

    @classmethod
    def from_config_json(cls, file: str) -> Self:
        """Alternative constructor of MambaConfig from a json configuration file."""
        with open(file, 'r') as f:
            config: dict[str, Any] = json.load(f)
        charset_file = config.get('charset')
        if not charset_file:
            charset = string.printable[:-3]
        else:
            with open(charset_file, 'r', encoding='utf-8') as cf:
                charset = cf.read()
        vocab_size = len(charset)
        d_model: int = config.get('d_model', 128)
        d_state: int = config.get('d_state', 8)
        expand_factor: float = config.get('expand_factor', 2.)
        d_conv: int = config.get('d_conv', 4)
        dt_rank: int | str = config.get('dt_rank', 'auto')
        n_layers: int = config.get('n_layers', 4)
        norm_eps: float = config.get('norm_eps', 1e-4)

        return cls(
            vocab_size,
            d_model,
            d_state,
            expand_factor,
            d_conv,
            dt_rank,
            n_layers,
            norm_eps,
            charset_file
        )



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

    def step(self, xt: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passo di inferenza autoregressivo per un singolo token.
        
        Args:
            xt: Input al timestep corrente. Shape: (Batch, d_inner)
            h:  Stato nascosto del timestep precedente. Shape: (Batch, d_inner, d_state)
                Se None, viene inizializzato a zeri.
                
        Returns:
            yt: Output per il timestep corrente. Shape: (Batch, d_inner)
            h:  Stato nascosto aggiornato da salvare in cache. Shape: (Batch, d_inner, d_state)
        """
        batch_size, d_inner = xt.shape
        device = xt.device

        # 1. Initialization of the state if it is the first token of the generation
        if h is None:
            h = torch.zeros(batch_size, d_inner, self.config.d_state, device=device)

        # 2. Time-Invariant A matrix
        A = -torch.exp(self.A_log.float())  # Shape: (d_inner, d_state)

        # 3. Parameters that depend on the current token (xt)
        dt = self.dt_proj(xt)               # (Batch, d_inner)
        dt = F.softplus(dt)                 # Grants positive time-step

        bt = self.B_proj(xt)                # (Batch, d_state)
        ct = self.C_proj(xt)                # (Batch, d_state)

        # 4. Discretization for a single time step
        # dA: (Batch, d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        
        # dB: (Batch, d_inner, d_state)
        dB = dt.unsqueeze(-1) * bt.unsqueeze(1)

        # 5. AUpdate of the latent state: h_t = dA * h_{t-1} + dB * x_t
        h = dA * h + dB * xt.unsqueeze(-1)

        # 6. Output computation: y_t = sum(h_t * C_t, dim=-1)
        yt = (h * ct.unsqueeze(1)).sum(dim=-1)  # (Batch, d_inner)

        # 7. Skip connection
        yt = yt + (xt * self.D)
        assert h is not None, "h at this point should be a tensor."
        return yt, h

    

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

    def step(self, x: torch.Tensor, 
             conv_state: torch.Tensor, 
             ssm_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            x: (Batch, d_model)
            conv_state: (Batch, d_inner, d_conv)
            ssm_state: (Batch, d_inner, d_state)
            """
            # 1. Input Projection
            x_proj = self.in_proj(x)
            x_main, x_gate = x_proj.chunk(2, dim=-1) # (Batch, d_inner)

            # 2. Causal 1D convoplution step-by-step
            # Left-Shift oif convolution state and insertion of the new token to the right
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = x_main
            
            # Manual computation of convolution, depthwise on the current buffer
            # self.conv1d.weight has shape (d_inner, 1, d_conv), squeezed becomes (d_inner, d_conv)
            x_main = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
            if self.conv1d.bias is not None:
                x_main = x_main + self.conv1d.bias
                
            x_main = F.silu(x_main)

            # 3. SSM Step
            x_main, ssm_state = self.ssm.step(x_main, ssm_state)

            # 4. Gating
            x_gate = F.silu(x_gate)
            x_combined = x_main * x_gate

            # 5. Output Projection
            out = self.out_proj(x_combined)
            
            return out, conv_state, ssm_state


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable parameter to scale the normalization (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Variance computation on the last dimension (d_model)
        variance = x.pow(2.).mean(-1, keepdim=True)
        # Normalization
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed