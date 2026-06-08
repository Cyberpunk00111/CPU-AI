"""Quick 300-step demo training for CogniCore -- produces a real checkpoint fast.

This script:
  1. Tokenises the first MAX_SAMPLES from the corpus (fast, ~2 min)
  2. Writes mini train.bin / val.bin shards
  3. Trains for MAX_STEPS steps and saves a checkpoint
  4. Prints final loss + perplexity

Run with:
    python scripts/quick_demo_train.py --config configs/phase1_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing
import sys
from pathlib import Path

# Make the project root importable so both cognicore and data packages resolve.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

# ── constants ────────────────────────────────────────────────────────────────
MAX_SAMPLES = 30_000   # already tokenised in data/demo/
MAX_STEPS   = 50       # 50 steps x ~3s = ~2.5 min on CPU
SEQ_LEN     = 128      # shorter context = faster forward pass for demo
BATCH_SIZE  = 8        # small batch for fast iteration
LOG_EVERY   = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))],
)
logger = logging.getLogger("quick_train")


def build_mini_shards(config_path: Path) -> tuple[Path, Path]:
    """Tokenise first MAX_SAMPLES corpus samples → mini train/val .bin shards."""
    from cognicore.model.config import load_config
    from cognicore.model.tokenizer.tokenizer_utils import load_tokenizer

    config = load_config(config_path)
    corpus = config.data.corpus_path
    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus}")

    tokenizer = load_tokenizer(config.data.tokenizer_path)
    logger.info("Tokenising first %d samples from corpus ...", MAX_SAMPLES)

    all_tokens: list[int] = []
    sample_buf: list[str] = []
    count = 0

    with corpus.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped == "<eos>":
                text = " ".join(sample_buf).strip()
                if text:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    all_tokens.extend(ids)
                    all_tokens.append(tokenizer.eos_id)
                sample_buf = []
                count += 1
                if count % 5000 == 0:
                    logger.info("  %d / %d samples | %d tokens", count, MAX_SAMPLES, len(all_tokens))
                if count >= MAX_SAMPLES:
                    break
            else:
                sample_buf.append(stripped)

    logger.info("Done: %d samples -> %d tokens", count, len(all_tokens))
    if not all_tokens:
        raise RuntimeError("No tokens produced -- check corpus.")

    split = int(len(all_tokens) * 0.9)
    train_arr = np.asarray(all_tokens[:split], dtype=np.uint16)
    val_arr   = np.asarray(all_tokens[split:], dtype=np.uint16)

    # Write to dedicated demo shards so full training isn't affected
    demo_dir = Path("data/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    train_path = demo_dir / "train.bin"
    val_path   = demo_dir / "val.bin"
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    logger.info("Wrote demo shards: train=%s val=%s", train_path, val_path)
    return train_path, val_path


def run_training(config_path: Path, train_path: Path, val_path: Path) -> Path:
    """Train for MAX_STEPS steps and return checkpoint path."""
    import math as _math
    import random as _random
    import numpy as _np
    import torch as _torch
    from cognicore.model.architecture import CogniCoreModel
    from cognicore.model.config import load_config
    from data.dataloader import TokenShardDataset
    from torch.utils.data import DataLoader

    def set_reproducibility(seed: int) -> None:
        _random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)

    def configure_cpu(cfg: object) -> None:
        if cfg.cpu.num_threads > 0:
            _torch.set_num_threads(cfg.cpu.num_threads)
        logger.info("CPU threads: %d", _torch.get_num_threads())

    def learning_rate_for_step(step: int, cfg: object) -> float:
        base_lr = cfg.training.learning_rate
        if step < cfg.training.warmup_steps:
            return base_lr * (step + 1) / max(cfg.training.warmup_steps, 1)
        progress = (step - cfg.training.warmup_steps) / max(
            cfg.training.max_steps - cfg.training.warmup_steps, 1
        )
        return base_lr * 0.5 * (1.0 + _math.cos(_math.pi * progress))

    config = load_config(config_path)
    set_reproducibility(config.training.seed)
    configure_cpu(config)

    # Override paths to use demo shards
    config.data.__dict__["train_path"] = train_path
    config.data.__dict__["val_path"]   = val_path

    model = CogniCoreModel(config)
    n_params = model.parameter_count()
    logger.info("Model: %s parameters (%.1f M)", f"{n_params:,}", n_params / 1e6)

    import torch
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )

    seq_len    = SEQ_LEN      # override to shorter context for speed
    batch_size = BATCH_SIZE

    train_ds = TokenShardDataset(train_path, seq_len)
    val_ds   = TokenShardDataset(val_path,   seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    ckpt_dir = Path("logs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    logger.info("Starting %d-step demo training (batch=%d seq=%d) ...", MAX_STEPS, batch_size, seq_len)

    for input_ids, targets in train_loader:
        if step >= MAX_STEPS:
            break
        lr = learning_rate_for_step(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, targets)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()

        if step % LOG_EVERY == 0:
            ppl = math.exp(min(out.loss.item(), 20))
            logger.info("step=%d  loss=%.4f  ppl=%.1f  lr=%.6f", step, out.loss.item(), ppl, lr)

        step += 1

    # Final validation loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for input_ids, targets in val_loader:
            out = model(input_ids, targets)
            if out.loss is not None:
                val_losses.append(float(out.loss.item()))
            if len(val_losses) >= 20:
                break
    val_loss = sum(val_losses) / max(len(val_losses), 1)
    val_ppl  = math.exp(min(val_loss, 20))
    logger.info("=" * 60)
    logger.info("Demo training complete!")
    logger.info("  Steps:      %d", step)
    logger.info("  Val loss:   %.4f", val_loss)
    logger.info("  Val ppl:    %.1f  (random baseline ~8192)", val_ppl)
    logger.info("=" * 60)

    ckpt_path = ckpt_dir / "cognicore_demo.pt"
    torch.save(
        {
            "step": step,
            "loss": val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.model_dump(mode="json"),
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved → %s", ckpt_path)
    return ckpt_path


def main() -> None:
    """CLI entry point."""
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Skip tokenisation if data/demo/train.bin already exists.")
    args = parser.parse_args()

    config_path = Path(args.config)
    demo_train = Path("data/demo/train.bin")

    if args.skip_tokenize and demo_train.exists():
        logger.info("Skipping tokenisation — using existing demo shards.")
        train_path = demo_train
        val_path   = Path("data/demo/val.bin")
    else:
        train_path, val_path = build_mini_shards(config_path)

    ckpt = run_training(config_path, train_path, val_path)
    logger.info("\nTo load this checkpoint in the API, set:")
    logger.info("  $env:COGNICORE_CONFIG     = '%s'", config_path.resolve())
    logger.info("  $env:COGNICORE_CHECKPOINT = '%s'", ckpt.resolve())


if __name__ == "__main__":
    main()
