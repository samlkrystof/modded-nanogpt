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
    position = torch.arange(max_seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    sincos_inp = position * div_term
    cos = torch.cos(sincos_inp)
    sin = torch.sin(sincos_inp)
    return cos, sin


class MultiheadLatentAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_dim: int, q_dim: int, rope_dim: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = kv_dim
        self.q_dim = q_dim
        self.rope_dim = rope_dim
        self.max_seq_len = max_seq_len

        self.rope_key = nn.Linear(d_model, rope_dim)
        self.rope_query = nn.Linear(q_dim, rope_dim)

        self.down_kv = nn.Linear(d_model, kv_dim)
        self.down_query = nn.Linear(d_model, q_dim)

        self.up_key = nn.Linear(kv_dim, d_model)
        self.up_value = nn.Linear(kv_dim, d_model)
        self.up_query = nn.Linear(q_dim, d_model)

        self.out = nn.Linear(d_model, d_model)

        # Precompute rotary embeddings at init time
        cos, sin = precompute_rotary_emb(max_seq_len, self.head_dim)
        self.register_buffer('rotary_emb', torch.stack([cos, sin]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        latent_kv = self.down_kv(x)
        latent_query = self.down_query(x)

        rope_key = self.rope_key(x)
        rope_key = rope_key.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        rope_query = self.rope_query(latent_query)
        rope_query = rope_query.view(batch_size, seq_len, self.n_heads, self.rope_dim)

        cos, sin = self.rotary_emb[0, :seq_len], self.rotary_emb[1, :seq_len]
        rope_query = apply_rotary_emb(rope_query, cos, sin, interleaved=True, inplace=False, seqlen_offsets=0)
        rope_key = apply_rotary_emb(rope_key, cos, sin, interleaved=True, inplace=False, seqlen_offsets=0)

        latent_query = self.up_query(latent_query)
        latent_query = latent_query.view(batch_size, seq_len, self.n_heads, self.head_dim)
        latent_key = self.up_key(latent_kv)
        latent_key = latent_key.view(batch_size, seq_len, self.n_heads, self.head_dim)
        latent_value = self.up_value(latent_kv)
        latent_value = latent_value.view(batch_size, seq_len, self.n_heads, self.head_dim)


        latent_query = torch.cat([latent_query, rope_query], dim=-1)
        latent_key = torch.cat([latent_key, rope_key], dim=-1)

        attn = F.scaled_dot_product_attention(
            latent_query.transpose(1, 2),
            latent_key.transpose(1, 2),
            latent_value.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        attn = self.out(attn)

        return attn

    

        


        






