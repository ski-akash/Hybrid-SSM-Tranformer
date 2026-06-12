import torch
import torch.nn as nn

def test_tensor_flow():
    # Define arbitrary dimensions
    batch_size, seq_len, d_model = 2, 16, 32
    
    # Create dummy input token embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Simple linear transformation to simulate a model block
    dummy_layer = nn.Linear(d_model, d_model)
    out = dummy_layer(x)
    
    print("--- Syntax Check Passed ---")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    test_tensor_flow()