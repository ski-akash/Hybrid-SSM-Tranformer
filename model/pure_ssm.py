import torch
import torch.nn as nn
from model.ssm_block import MambaBlock

class PureSSM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Only SSM Layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        
        for ssm, norm in zip(self.layers, self.norms):
            residual = x
            x = ssm(norm(x)) + residual
            
        x = self.final_norm(x)
        return self.output_head(x)