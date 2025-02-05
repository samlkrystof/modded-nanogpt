import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .rotary import apply_rotary_emb

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

class TensorProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, q_rank: int, kv_rank: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_rank = q_rank
        self.kv_rank = kv_rank

        self.aq_proj = nn.Linear(d_model, q_rank * n_heads)
        self.bq_proj = nn.Linear(d_model, q_rank * self.head_dim)

        self.ak_proj = nn.Linear(d_model, kv_rank * n_heads)
        self.bk_proj = nn.Linear(d_model, kv_rank * self.head_dim)

        self.av_proj = nn.Linear(d_model, kv_rank * n_heads)
        self.bv_proj = nn.Linear(d_model, kv_rank * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

        self.cos, self.sin = precompute_rotary_emb(max_seq_len, self.head_dim)

    def _init_weights(self):
        aq_tensor = self.aq_proj.weight.view(self.d_model, self.n_heads, self.q_rank)
        ak_tensor = self.ak_proj.weight.view(self.d_model, self.n_heads, self.kv_rank)
        av_tensor = self.av_proj.weight.view(self.d_model, self.n_heads, self.kv_rank)
        nn.init.xavier_uniform_(aq_tensor)
        nn.init.xavier_uniform_(ak_tensor)
        nn.init.xavier_uniform_(av_tensor)
        self.aq_proj.weight.data = aq_tensor.view_as(self.aq_proj.weight)
        self.ak_proj.weight.data = ak_tensor.view_as(self.ak_proj.weight)
        self.av_proj.weight.data = av_tensor.view_as(self.av_proj.weight)

        bq_tensor = self.bq_proj.weight.view(self.d_model, self.q_rank, self.head_dim)
        bk_tensor = self.bk_proj.weight.view(self.d_model, self.kv_rank, self.head_dim)
        bv_tensor = self.bv_proj.weight.view(self.d_model, self.kv_rank, self.head_dim)
        nn.init.xavier_uniform_(bq_tensor)
        nn.init.xavier_uniform_(bk_tensor)
        nn.init.xavier_uniform_(bv_tensor)
        self.bq_proj.weight.data = bq_tensor.view_as(self.bq_proj.weight)
        self.bk_proj.weight.data = bk_tensor.view_as(self.bk_proj.weight)
        self.bv_proj.weight.data = bv_tensor.view_as(self.bv_proj.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        a_q = self.aq_proj(x).view(batch_size, seq_len, self.n_heads, self.q_rank)
        a_k = self.ak_proj(x).view(batch_size, seq_len, self.n_heads, self.kv_rank)
        a_v = self.av_proj(x).view(batch_size, seq_len, self.n_heads, self.kv_rank)

        b_q = self.bq_proj(x).view(batch_size, seq_len, self.q_rank, self.head_dim)
        b_k = self.bk_proj(x).view(batch_size, seq_len, self.kv_rank, self.head_dim)
        b_v = self.bv_proj(x).view(batch_size, seq_len, self.kv_rank, self.head_dim)

        b_q = apply_rotary_emb(b_q, self.cos, self.sin, interleaved=True, inplace=False, seqlen_offsets=0)
        b_k = apply_rotary_emb(b_k, self.cos, self.sin, interleaved=True, inplace=False, seqlen_offsets=0)

        query = a_q @ b_q
        key = a_k @ b_k
        value = a_v @ b_v

        attn = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        attn = self.out_proj(attn)

        return attn

        
