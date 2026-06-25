"""Download and combine instruction-only datasets for CogniCore SFT."""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset

from cognicore.model.config import CogniCoreConfig, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cognicore.sft.download")


def _dataset_specs() -> list[dict[str, object]]:
    """Return only instruction/conversation datasets (no Wikipedia)."""
    return [
        {
            "name": "OpenAssistant/oasst2",
            "config": None,
            "split": "train",
            "text_column": "text",
            "max_samples": None,
        },
        {
            "name": "databricks/databricks-dolly-15k",
            "config": None,
            "split": "train",
            "text_column": "instruction",
            "max_samples": None,
        },
    ]


def _normalise_record(record: dict[str, object], text_column: str) -> str:
    """Convert a dataset row into a training text sample."""
    if "instruction" in record and "response" in record:
        instruction = str(record.get("instruction", "")).strip()
        response = str(record.get("response", "")).strip()
        return f"{instruction}\n{response}".strip()
    return str(record.get(text_column, "")).strip()


def iter_samples() -> Iterable[str]:
    """Yield text samples from instruction datasets."""
    for spec in _dataset_specs():
        logger.info("Loading instruction dataset %s", spec["name"])
        try:
            dataset = load_dataset(
                str(spec["name"]),
                str(spec["config"]) if spec["config"] else None,
                split=str(spec["split"]),
                streaming=True,
            )
            max_samples = spec["max_samples"]
            for index, row in enumerate(dataset):
                if max_samples is not None and index >= int(max_samples):
                    break
                text = _normalise_record(dict(row), str(spec["text_column"]))
                if text:
                    yield text
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", spec["name"], e)
            raise e


def build_sft_corpus(config_path: str | Path) -> Path:
    """Download, shuffle, and write the instruction corpus."""
    config = load_config(config_path)
    logger.info("Fetching instruction samples...")
    samples = list(iter_samples())
    
    logger.info("Shuffling %d samples with seed %d...", len(samples), config.training.seed)
    rng = random.Random(config.training.seed)
    rng.shuffle(samples)

    config.data.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with config.data.corpus_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.replace("\r\n", "\n"))
            handle.write("\n<eos>\n")

    logger.info("Wrote %d SFT samples to %s", len(samples), config.data.corpus_path)
    return config.data.corpus_path


def main() -> None:
    """CLI entry point for SFT dataset download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to SFT YAML config.")
    args = parser.parse_args()
    build_sft_corpus(args.config)


if __name__ == "__main__":
    main()
