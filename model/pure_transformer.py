import torch
import torch.nn as nn
from model.attention_block import MultiHeadAttention

class PureTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Only Attention Layers
        self.layers = nn.ModuleList([
            MultiHeadAttention(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.shape[1]
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = attn(norm(x), mask=causal_mask) + residual
            
        x = self.final_norm(x)
        return self.output_head(x)