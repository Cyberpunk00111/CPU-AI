"""Tests for training utilities."""

from cognicore.model.config import load_config
from cognicore.model.training import learning_rate_for_step


def test_learning_rate_warmup_increases() -> None:
    """Warmup learning rate should increase between the first two steps."""
    config = load_config("configs/phase1_config.yaml")
    assert learning_rate_for_step(1, config) > learning_rate_for_step(0, config)
