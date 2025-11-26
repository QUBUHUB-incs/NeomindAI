import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------
# Rotary Positional Embeddings (RoPE) with scaling
# ---------------------------------------
def apply_rope(q, k, seq_len, rope_theta=10000.0, rope_scaling=None):
    """
    Apply RoPE to queries and keys. Supports long-context scaling.
    q, k: (B, H, T, D)
    rope_scaling: dict with type='longrope', short_factor, long_factor
    """
    B, H, T, D = q.shape
    dim = D * 2  # RoPE typically splits heads in 2
    theta = torch.arange(0, dim, 2, device=q.device, dtype=q.dtype) / dim
    theta = 1.0 / (rope_theta ** theta)
    pos = torch.arange(T, device=q.device, dtype=q.dtype)

    freqs = torch.einsum('i,j->ij', pos, theta)  # (T, D/2)

    if rope_scaling is not None:
        short_factor = torch.tensor(rope_scaling['short_factor'], device=q.device, dtype=q.dtype)
        long_factor = torch.tensor(rope_scaling['long_factor'], device=q.device, dtype=q.dtype)
        # linear interpolation scaling for simplicity
        freqs = freqs * short_factor + freqs * long_factor

    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    
    # split last dim
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q, k

# ---------------------------------------
# Sliding window attention for long sequences
# ---------------------------------------
def sliding_window_attention(q, k, v, window_size):
    """
    Computes local attention with a sliding window.
    q, k, v: (B, H, T, D)
    window_size: int
    """
    B, H, T, D = q.shape
    context = torch.zeros_like(q)
    
    for i in range(T):
        start = max(0, i - window_size // 2)
        end = min(T, i + window_size // 2 + 1)
        attn_scores = torch.matmul(q[:, :, i:i+1, :], k[:, :, start:end, :].transpose(-2, -1)) / math.sqrt(D)
        attn_probs = F.softmax(attn_scores, dim=-1)
        context[:, :, i:i+1, :] = torch.matmul(attn_probs, v[:, :, start:end, :])
    return context

# ---------------------------------------
# NeoMind Attention with RoPE + Sliding Window
# ---------------------------------------
class NeoMindAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = config.sliding_window

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling

    def forward(self, x, mask=None, past_kv=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)

        # Apply RoPE
        q, k = apply_rope(q, k, seq_len=T, rope_theta=self.rope_theta, rope_scaling=self.rope_scaling)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        # Sliding window or full attention
        if self.window_size is not None:
            context = sliding_window_attention(q, k, v, self.window_size)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            context = torch.matmul(attn_probs, v)

        out = self.resid_dropout(self.out_proj(context.transpose(1,2).contiguous().view(B, T, C)))
        return out, (k, v)
