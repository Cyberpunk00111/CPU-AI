"""Export a trained CogniCore checkpoint to ONNX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from cognicore.model.architecture import CogniCoreModel
from cognicore.model.config import load_config

logger = logging.getLogger(__name__)


def export_onnx(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export CogniCore logits graph to ONNX.

    Args:
        config_path: Path to CogniCore YAML config.
        checkpoint_path: Path to a PyTorch checkpoint.
        output_path: Destination ONNX path.

    Returns:
        The ONNX output path.
    """
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = CogniCoreModel(config)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    dummy = torch.zeros(1, config.model.max_seq_length, dtype=torch.long)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        destination,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "sequence"}, "logits": {1: "sequence"}},
        opset_version=17,
    )
    logger.info("Exported ONNX model to %s", destination)
    return destination


def main() -> None:
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    export_onnx(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
