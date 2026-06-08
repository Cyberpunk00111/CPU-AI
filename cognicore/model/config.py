"""Configuration loading for CogniCore.

The project is intentionally configuration-driven: model dimensions, training
hyperparameters, paths, and CPU switches live in YAML and are validated here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal local runtimes.
    yaml = None


class ModelConfig(BaseModel):
    """Validated transformer architecture settings."""

    vocab_size: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    ffn_hidden_dim: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    weight_bits: float = Field(gt=0.0)
    activation_bits: int = Field(gt=0)
    norm_eps: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_attention_shape(self) -> ModelConfig:
        """Ensure attention dimensions are compatible with grouped-query attention."""
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        return self


class TrainingConfig(BaseModel):
    """Validated training hyperparameters."""

    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    beta1: float = Field(gt=0.0, lt=1.0)
    beta2: float = Field(gt=0.0, lt=1.0)
    warmup_steps: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    grad_clip: float = Field(gt=0.0)
    save_every: int = Field(gt=0)
    eval_every: int = Field(gt=0)
    seed: int


class DataConfig(BaseModel):
    """Validated data and tokenizer paths."""

    train_path: Path
    val_path: Path
    raw_dir: Path
    corpus_path: Path
    num_workers: int = Field(ge=0)
    tokenizer_path: Path


class CPUConfig(BaseModel):
    """Validated CPU execution switches."""

    num_threads: int = Field(ge=0)
    use_gradient_checkpointing: bool
    compile: bool


class LoggingConfig(BaseModel):
    """Validated logging settings."""

    log_dir: Path
    wandb: bool
    log_every: int = Field(gt=0)


class CogniCoreConfig(BaseModel):
    """Top-level CogniCore configuration."""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    cpu: CPUConfig
    logging: LoggingConfig


def load_config(path: str | Path) -> CogniCoreConfig:
    """Load and validate a CogniCore YAML config.

    Args:
        path: Filesystem path to a YAML config.

    Returns:
        A validated CogniCoreConfig instance.

    Raises:
        FileNotFoundError: If the config path does not exist.
        ValueError: If YAML parsing or Pydantic validation fails.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        text = config_path.read_text(encoding="utf-8")
        raw_config = _parse_yaml(text)
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Failed to parse YAML config at %s", config_path)
        raise ValueError(f"Invalid YAML config at {config_path}: {exc}") from exc

    try:
        config = CogniCoreConfig.model_validate(raw_config)
    except ValidationError as exc:
        logger.exception("Config validation failed for %s", config_path)
        raise ValueError(f"Invalid CogniCore config at {config_path}: {exc}") from exc

    logger.info("Loaded CogniCore config from %s", config_path)
    return config


def _parse_yaml(text: str) -> dict[str, Any]:
    """Parse project YAML with PyYAML when available and a small fallback otherwise."""
    if yaml is not None:
        return yaml.safe_load(text)
    parsed: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        if not line_without_comment.startswith(" ") and line_without_comment.endswith(":"):
            section_name = line_without_comment[:-1]
            current_section = {}
            parsed[section_name] = current_section
            continue
        if current_section is None or ":" not in line_without_comment:
            raise ValueError("Fallback YAML parser only supports one-level mapping sections")
        key, raw_value = line_without_comment.strip().split(":", 1)
        current_section[key.strip()] = _parse_scalar(raw_value.strip())
    return parsed


def _parse_scalar(value: str) -> str | int | float | bool:
    """Parse the scalar values used in CogniCore config files."""
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(marker in cleaned for marker in (".", "e", "E")):
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned
