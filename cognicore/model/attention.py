"""Grouped-query self-attention with Rotary Position Embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from cognicore.model.bitlinear import BitLinear


class RotaryEmbedding(nn.Module):
    """Precomputed rotary embedding frequencies for attention heads.

    Args:
        head_dim: Size of each attention head.
        max_seq_length: Maximum context length.
        base: RoPE frequency base.
    """

    def __init__(self, head_dim: int, max_seq_length: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_length, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply RoPE to a query or key tensor.

        Args:
            x: Tensor shaped (batch, heads, sequence, head_dim).
            start_pos: Absolute position offset for cached decoding.

        Returns:
            Rotated tensor with the same shape as x.
        """
        seq_len = x.size(-2)
        cos = self.cos[start_pos : start_pos + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[start_pos : start_pos + seq_len].unsqueeze(0).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    """Causal grouped-query attention with BitLinear projections.

    Args:
        embedding_dim: Transformer hidden size.
        num_heads: Number of query heads.
        num_kv_heads: Number of shared key/value heads.
        max_seq_length: Maximum context length.
        activation_bits: Activation quantisation bit width.
        dropout: Attention and residual dropout probability.
        eps: Numerical stability constant for BitLinear.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_length: int,
        activation_bits: int,
        dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embedding_dim // num_heads
        self.kv_repeats = num_heads // num_kv_heads
        self.q_proj = BitLinear(embedding_dim, embedding_dim, activation_bits, False, eps)
        self.k_proj = BitLinear(
            embedding_dim,
            num_kv_heads * self.head_dim,
            activation_bits,
            False,
            eps,
        )
        self.v_proj = BitLinear(
            embedding_dim,
            num_kv_heads * self.head_dim,
            activation_bits,
            False,
            eps,
        )
        self.o_proj = BitLinear(embedding_dim, embedding_dim, activation_bits, False, eps)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_length)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _shape_heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        """Reshape projection output to attention-head layout."""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, heads, self.head_dim).transpose(1, 2)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat key/value heads so they align with query heads."""
        if self.kv_repeats == 1:
            return x
        return x.repeat_interleave(self.kv_repeats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal grouped-query self-attention.

        Args:
            x: Hidden states shaped (batch, sequence, embedding_dim).

        Returns:
            Attention output with the same shape as x.
        """
        batch_size, seq_len, _ = x.shape
        q = self.rope(self._shape_heads(self.q_proj(x), self.num_heads))
        k = self.rope(self._shape_heads(self.k_proj(x), self.num_kv_heads))
        v = self._shape_heads(self.v_proj(x), self.num_kv_heads)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        attn_scores = attn_scores.masked_fill(~causal_mask, torch.finfo(attn_scores.dtype).min)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        return self.resid_dropout(self.o_proj(out))
