import torch
import torch.nn as nn
from model.ssm_block import MambaBlock
from model.attention_block import MultiHeadAttention

class HybridTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_ssm_layers, num_attn_layers, num_heads):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Hybrid Blocks - Interleaving Strategy
        # Early layers use SSM blocks for efficient local pattern capture
        self.ssm_layers = nn.ModuleList([
            MambaBlock(d_model=d_model) for _ in range(num_ssm_layers)
        ])
        
        # Later layers use attention blocks for global dependency modeling
        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(d_model=d_model, num_heads=num_heads) for _ in range(num_attn_layers)
        ])
        
        # 3. Pre-Norm Layer Normalization for stability
        self.ssm_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_ssm_layers)])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_attn_layers)])
        
        # 4. Final Layer Norm and Output Head (predicts next token)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length) - contains integer token IDs
        """
        # Convert token IDs to embeddings
        x = self.embedding(x)
        
        # Pass through early SSM layers with residual connections
        for ssm, norm in zip(self.ssm_layers, self.ssm_norms):
            residual = x
            x = ssm(norm(x)) + residual
            
        # Create a causal mask for the attention layers
        seq_len = x.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        
        # Pass through later Attention layers with residual connections
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            residual = x
            x = attn(norm(x), mask=causal_mask) + residual
            
        # Final normalization and projection to vocabulary size
        x = self.final_norm(x)
        logits = self.output_head(x)
        
        return logits

# --- Verification Block ---
if __name__ == "__main__":
    BATCH_SIZE = 2
    SEQ_LEN = 64
    VOCAB_SIZE = 1000
    D_MODEL = 128
    NUM_HEADS = 8
    NUM_SSM_LAYERS = 2
    NUM_ATTN_LAYERS = 2
    
    print("Initializing Hybrid SSM-Transformer...")
    model = HybridTransformer(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        num_ssm_layers=NUM_SSM_LAYERS, 
        num_attn_layers=NUM_ATTN_LAYERS, 
        num_heads=NUM_HEADS
    )
    
    # Create dummy integer inputs representing token IDs
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    print(f"Input shape: {x.shape} -> (Batch, Seq_Len)")
    
    print("Running forward pass through full hybrid architecture...")
    logits = model(x)
    
    print(f"Output shape: {logits.shape} -> (Batch, Seq_Len, Vocab_Size)")
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), "Error: Output shape is incorrect!"
    print("✅ Success! The Hybrid Model successfully integrated Mamba and Attention blocks.")