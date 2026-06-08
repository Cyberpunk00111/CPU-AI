"""Full Phase 1 training pipeline for CogniCore — Windows-compatible runner.

Runs all four pipeline stages in sequence:
  1. Tokenizer training  (skipped if tokenizer.json already exists)
  2. Corpus tokenisation / preprocessing  (skipped if train.bin already exists)
  3. Model training  (50 000 steps, checkpoints every 500 steps)

Usage
-----
    python scripts/run_phase1.py --config configs/phase1_config.yaml

Optional flags
--------------
    --skip-tokenizer   Skip tokenizer training (use existing files)
    --skip-preprocess  Skip corpus preprocessing (use existing .bin files)
    --skip-training    Only run tokenizer + preprocessing, skip model training
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — human-readable timestamps, level, and module name
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cognicore.pipeline")


def _banner(title: str) -> None:
    """Print a section banner to stdout."""
    line = "=" * 70
    logger.info(line)
    logger.info("  %s", title)
    logger.info(line)


def stage_tokenizer(config_path: Path, skip: bool) -> None:
    """Train BPE tokenizer from corpus — skip if already done."""
    from cognicore.model.config import load_config

    config = load_config(config_path)
    tokenizer_json = config.data.tokenizer_path / "tokenizer.json"

    if skip or tokenizer_json.exists():
        logger.info(
            "Tokenizer already exists at %s — skipping training.", tokenizer_json
        )
        return

    _banner("STAGE 1 — Training BPE tokenizer")
    if not config.data.corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {config.data.corpus_path}\n"
            "Run data/download.py first."
        )

    from cognicore.model.tokenizer.train_tokenizer import train_tokenizer

    out = train_tokenizer(config_path)
    logger.info("Tokenizer saved to %s", out)


def stage_preprocess(config_path: Path, skip: bool) -> None:
    """Tokenise corpus to binary shards — skip if already done."""
    from cognicore.model.config import load_config

    config = load_config(config_path)

    if skip or config.data.train_path.exists():
        logger.info(
            "Binary shards already exist at %s — skipping preprocessing.",
            config.data.train_path,
        )
        return

    _banner("STAGE 2 — Tokenising corpus to binary shards")
    from data.preprocess import preprocess

    train_path, val_path = preprocess(config_path)
    logger.info("Train shard: %s", train_path)
    logger.info("Val shard:   %s", val_path)


def stage_training(config_path: Path, skip: bool) -> None:
    """Run the full training loop."""
    if skip:
        logger.info("Training skipped (--skip-training flag).")
        return

    _banner("STAGE 3 — Phase 1 Training (50 000 steps)")
    from cognicore.model.training import Trainer, load_config

    config = load_config(config_path)
    trainer = Trainer(config)
    logger.info(
        "Model parameters: %s",
        f"{trainer.model.parameter_count():,}",
    )
    trainer.train()
    logger.info("Training complete.")


def main() -> None:
    """Parse CLI args and run the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to CogniCore YAML config (e.g. configs/phase1_config.yaml).",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer training and use existing tokenizer files.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip corpus preprocessing and use existing .bin shards.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (run only tokenizer + preprocessing).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        stage_tokenizer(config_path, skip=args.skip_tokenizer)
        stage_preprocess(config_path, skip=args.skip_preprocess)
        stage_training(config_path, skip=args.skip_training)
        logger.info("Pipeline finished successfully.")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
