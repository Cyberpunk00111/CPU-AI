"""Run the complete Phase 1 data, tokenizer, preprocessing, and training pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cognicore.model.config import load_config
from cognicore.model.tokenizer.train_tokenizer import train_tokenizer
from cognicore.model.training import Trainer
from data.download import build_corpus
from data.preprocess import preprocess


def run_pipeline(config_path: str | Path) -> None:
    """Execute the full Phase 1 pipeline in dependency order.

    Args:
        config_path: Path to the CogniCore YAML config.
    """
    config = load_config(config_path)
    config.logging.log_dir.mkdir(parents=True, exist_ok=True)

    if not config.data.corpus_path.exists():
        build_corpus(config_path)
    else:
        logging.info("Skipping corpus download; found %s", config.data.corpus_path)

    tokenizer_file = config.data.tokenizer_path / "tokenizer.json"
    if not tokenizer_file.exists():
        train_tokenizer(config_path)
    else:
        logging.info("Skipping tokenizer training; found %s", tokenizer_file)

    if not config.data.train_path.exists() or not config.data.val_path.exists():
        preprocess(config_path)
    else:
        logging.info(
            "Skipping preprocessing; found %s and %s",
            config.data.train_path,
            config.data.val_path,
        )

    Trainer(config).train()


def main() -> None:
    """CLI entry point for the full Phase 1 pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase1_config.yaml",
        help="Path to CogniCore YAML config.",
    )
    args = parser.parse_args()
    log_path = Path("logs") / "phase1_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
