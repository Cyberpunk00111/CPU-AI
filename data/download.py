"""Download and combine open Phase 1 datasets for CogniCore."""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset

from cognicore.model.config import CogniCoreConfig, load_config

logger = logging.getLogger(__name__)


def _dataset_specs(config: CogniCoreConfig) -> list[dict[str, object]]:
    """Return reproducible Phase 1 dataset specs."""
    return [
        {
            "name": "wikimedia/wikipedia",
            "config": "20231101.en",
            "split": "train",
            "text_column": "text",
            "max_samples": 500_000,
        },
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


def iter_samples(config: CogniCoreConfig) -> Iterable[str]:
    """Yield text samples from configured open datasets."""
    for spec in _dataset_specs(config):
        logger.info("Loading dataset %s", spec["name"])
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


def build_corpus(config_path: str | Path) -> Path:
    """Download, shuffle, and write the configured Phase 1 corpus.

    Args:
        config_path: Path to CogniCore YAML config.

    Returns:
        Path to the combined corpus file.
    """
    config = load_config(config_path)
    samples = list(iter_samples(config))
    rng = random.Random(config.training.seed)
    rng.shuffle(samples)

    config.data.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with config.data.corpus_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.replace("\r\n", "\n"))
            handle.write("\n<eos>\n")

    logger.info("Wrote %d samples to %s", len(samples), config.data.corpus_path)
    return config.data.corpus_path


def main() -> None:
    """CLI entry point for dataset download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to CogniCore YAML config.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_corpus(args.config)


if __name__ == "__main__":
    main()
