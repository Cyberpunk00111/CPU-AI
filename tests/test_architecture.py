"""Tests for the full CogniCore transformer."""

import torch

from cognicore.model.architecture import CogniCoreModel
from cognicore.model.config import load_config


def test_model_forward_loss_shape() -> None:
    """The model should return logits and scalar cross-entropy loss."""
    config = load_config("configs/phase1_config.yaml")
    model = CogniCoreModel(config)
    input_ids = torch.randint(0, config.model.vocab_size, (2, 8))
    output = model(input_ids, targets=input_ids)
    assert output.logits.shape == (2, 8, config.model.vocab_size)
    assert output.loss is not None
    assert output.loss.ndim == 0
