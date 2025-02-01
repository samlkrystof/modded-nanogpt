import math
import torch
import torch.nn.functional as F
from torch import nn

from .rotary import apply_rotary_emb
from flex_head_fa import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from .rms_norm import RMSNorm


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


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


class MultiheadFlashDiff(nn.Module):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized flex_head_fa (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """
    def __init__(
        self,
        args,
        embed_dim,
        depth,
        num_heads,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of Transformer's num_heads
        self.num_heads = num_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        
        # Add these lines to store max_seq_len
        self.max_seq_len = getattr(args, 'max_seq_len', 2048)  # default to 2048 if not specified
        
        # Precompute rotary embeddings at init time
        self.register_buffer(
            'rotary_emb',
            torch.stack(
                precompute_rotary_emb(
                    max_seq_len=self.max_seq_len,
                    dim=self.head_dim,
                    device=self.lambda_q1.device
                )
            ),
            persistent=False
        )
    
    def forward(
        self,
        x,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, 2 * self.head_dim)

        # Get the relevant slice of our precomputed embeddings
        cos, sin = self.rotary_emb[0, :tgt_len], self.rotary_emb[1, :tgt_len]

        # Apply rotary embeddings with correct parameters
        q = apply_rotary_emb(
            q,
            cos,
            sin,
            interleaved=True,
            inplace=False,
            seqlen_offsets=0,
        )
        k = apply_rotary_emb(
            k,
            cos,
            sin,
            interleaved=True,
            inplace=False,
            seqlen_offsets=0,
        )

        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        attn1 = flash_attn_func(q1, k1, v, causal=True)
        attn2 = flash_attn_func(q2, k2, v, causal=True)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        attn = self.out_proj(attn)
        return attn