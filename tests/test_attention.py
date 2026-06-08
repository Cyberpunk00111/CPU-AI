"""Tests for grouped-query attention."""

import torch

from cognicore.model.attention import GroupedQueryAttention


def test_grouped_query_attention_shape() -> None:
    """GQA should preserve hidden shape."""
    attention = GroupedQueryAttention(
        embedding_dim=32,
        num_heads=4,
        num_kv_heads=2,
        max_seq_length=16,
        activation_bits=8,
        dropout=0.0,
        eps=1.0e-6,
    )
    output = attention(torch.randn(2, 8, 32))
    assert output.shape == (2, 8, 32)
