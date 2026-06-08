"""Tests for CogniCore configuration loading."""

from cognicore.model.config import load_config


def test_load_phase1_config() -> None:
    """Phase 1 config loads and validates GQA dimensions."""
    config = load_config("configs/phase1_config.yaml")
    assert config.model.vocab_size == 8192
    assert config.model.num_heads % config.model.num_kv_heads == 0
