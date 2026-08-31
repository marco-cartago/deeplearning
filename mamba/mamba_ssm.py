import torch
import torch.nn as nn

from mamba import MambaConfig, MambaBlock, RMSNorm


class MambaLayer(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        # Layer normalization before SSM
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ssm = MambaBlock(config) 
        
        # LayerNorm before MLP
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, x):
        """
        x shape: (Batch, Sequence_Length, d_model)
        """
        # SSM branch with residual connection (Pre-Norm)
        residual = x
        x = self.norm1(x)
        x = self.ssm(x)
        x = x + residual
        
        # MLP branch with residual connection (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = x + residual
        
        return x

    def step(self, x: torch.Tensor, 
             cache: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        conv_state, ssm_state = cache
        
        # SSM branch with residual connection (Pre-Norm)
        residual = x
        x = self.norm1(x)
        x, conv_state, ssm_state = self.ssm.step(x, conv_state, ssm_state)
        x = x + residual
        
        # MLP branch with residual connection (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = x + residual
        
        return x, (conv_state, ssm_state)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Stack some MambaLayers using ModuleList
        self.layers = nn.ModuleList([
            MambaLayer(config) for _ in range(config.n_layers)
        ])
        
        # Normalizzazione finale prima dell'output (standard per i decoder-only)
        # self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, x):
        """
        x shape: (Batch, Sequence_Length, d_model)
        Output shape: (Batch, Sequence_Length, d_model)
        """

        for layer in self.layers:
            x = layer(x)
            
        # Applica la normalizzazione finale sulle features estratte
        # x = self.final_norm(x)
        return x

    def step(self, x: torch.Tensor, caches: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, new_cache = layer.step(x, caches[i]) # type: ignore[operator]
            new_caches.append(new_cache)
        return x, new_caches
    

class MambaForCausalLM(nn.Module):
    """
    Wraps the Mamba backbone with:
      - a token embedding (int token ids -> d_model vectors)
      - an LM head (d_model -> vocab logits), tied to the embedding weights
 
    This is what was missing before: previously `Mamba` was fed raw
    integer input_ids directly, which cannot work (in_proj etc. expect
    float vectors of size d_model, not token indices), and there was no
    way to get vocab-sized logits back out.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.backbone = Mamba(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
 
        # Weight tying between the embedding and the output projection is
        # standard practice for LMs like this: it saves a large chunk of
        # parameters (the embedding/head is usually the biggest matrix in
        # a small model) and tends to help small models generalize.
        self.lm_head.weight = self.embedding.weight
 
    def forward(self, input_ids):
        """
        input_ids shape: (Batch, Sequence_Length), dtype long
        Output (logits) shape: (Batch, Sequence_Length, vocab_size)
        """
        x = self.embedding(input_ids)
        x = self.backbone(x)
        logits = self.lm_head(x)
        return logits

    def allocate_caches(self, 
                        batch_size: int, 
                        device: str | torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Initialize to zero the convolution and SSM states for all layers."""
        caches = []
        for _ in range(self.config.n_layers):
            conv_state = torch.zeros(batch_size, self.config.d_inner, self.config.d_conv, device=device)
            ssm_state = torch.zeros(batch_size, self.config.d_inner, self.config.d_state, device=device)
            caches.append((conv_state, ssm_state))
        return caches

    def step(self, input_id: torch.Tensor, caches: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, list]:
        """
        input_id shape: (Batch,) oppure (Batch, 1)
        Retruns logits for a single next token and the new cached values.
        """
        x = self.embedding(input_id)
        if x.dim() == 3: # Se è passata un'extra dim per Sequence_Length=1, la rimuoviamo
            x = x.squeeze(1)
            
        x, new_caches = self.backbone.step(x, caches)
        logits = self.lm_head(x) # (Batch, vocab_size)
        return logits, new_caches