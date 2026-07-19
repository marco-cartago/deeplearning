import torch
import torch.nn as nn

from mamba import MambaConfig, MambaBlock, RMSNorm, MLP


class MambaLayer(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        # Normalizzazione prima del blocco SSM
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        # Assumiamo che MambaBlock sia la classe creata nel passaggio precedente
        self.ssm = MambaBlock(config) 
        
        # Normalizzazione prima del blocco MLP
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        x shape: (Batch, Sequence_Length, d_model)
        """
        # Ramo SSM con connessione residuale (Pre-Norm)
        residual = x
        x = self.norm1(x)
        x = self.ssm(x)
        x = x + residual
        
        # Ramo MLP con connessione residuale (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Accatasta un numero arbitrario di MambaLayer usando nn.ModuleList
        self.layers = nn.ModuleList([
            MambaLayer(config) for _ in range(config.n_layers)
        ])
        
        # Normalizzazione finale prima dell'output (standard per i decoder-only)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, x):
        """
        x shape: (Batch, Sequence_Length, d_model)
        Output shape: (Batch, Sequence_Length, d_model)
        """
        # Passa attraverso l'intero stack di layer sequenzialmente
        for layer in self.layers:
            x = layer(x)
            
        # Applica la normalizzazione finale sulle features estratte
        x = self.final_norm(x)
        return x
    

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
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
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