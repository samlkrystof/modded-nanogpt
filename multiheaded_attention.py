import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotary import apply_rotary_emb
import math

def precompute_rotary_emb(max_seq_len: int, dim: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute rotary embeddings for the entire model.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Dimension of the embeddings (usually head_dim)
        device: Device to put the embeddings on
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: cos and sin embeddings
    """
    # Compute position indices
    position = torch.arange(max_seq_len, device=device).unsqueeze(1)
    # Compute division term
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    # Compute sin and cos embeddings
    sincos_inp = position * div_term
    cos = torch.cos(sincos_inp)
    sin = torch.sin(sincos_inp)
    return cos, sin



class MultiheadAttention(nn.Module):
    """
    Standard Multihead Attention with RoPE (Rotary Position Embeddings).
    Uses PyTorch's scaled_dot_product_attention for efficient computation.
    """
    def __init__(
        self,
        args,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = dropout
        
        # Store max_seq_len from args or default
        self.max_seq_len = getattr(args, 'max_seq_len', 2048)
        
        # Precompute rotary embeddings at init time
        self.register_buffer(
            'rotary_emb',
            torch.stack(
                precompute_rotary_emb(
                    max_seq_len=self.max_seq_len,
                    dim=self.head_dim,
                    device='cpu'  # Will be moved to correct device when model is moved
                )
            ),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        bsz, tgt_len, _ = x.size()
        
        # Project to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim)
        
        # Get rotary embeddings for current sequence length
        cos, sin = self.rotary_emb[0, :tgt_len], self.rotary_emb[1, :tgt_len]
        
        # Apply rotary embeddings
        q = apply_rotary_emb(q, cos, sin, interleaved=True, inplace=False)
        k = apply_rotary_emb(k, cos, sin, interleaved=True, inplace=False)
        
        # Reshape for scaled_dot_product_attention
        # (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply scaled dot product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # Causal mask is automatically applied when needed
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back to (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        return output