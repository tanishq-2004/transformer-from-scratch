"""Training loop for the from-scratch Transformer.

Usage:
    python -m src.train --config configs/small.yaml
    python -m src.train --config configs/smoke.yaml --max_steps 50

What it does:
  - loads a YAML config (see configs/small.yaml)
  - trains a BPE tokenizer if tokenizer.json is missing, otherwise loads it
  - loads TinyStories (or Tiny Shakespeare for the smoke config) and
    encodes it once into a flat token array
  - trains with AdamW (lr 3e-4, betas 0.9/0.95, weight decay 0.1 on
    matrices only), 500-step warmup then cosine decay, grad clipping 1.0,
    cross-entropy loss, mixed precision on CUDA (bf16) when available
  - evaluates train/val loss every `eval_interval` steps, logs train loss,
    val loss and perplexity (exp(val loss)) to logs/train_log.csv and
    logs/train_log.json, checkpoints every `checkpoint_interval` steps

All numbers written to the log come from the actual run. Nothing is invented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import torch
import yaml

from src.data import (
    download_tiny_shakespeare,
    encode_texts_to_array,
    get_batch,
    load_tinystories_texts,
)
from src.model import TransformerConfig, TransformerLM
from src.tokenizer import encode, load_tokenizer, train_bpe_tokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_lr(step: int, cfg: dict) -> float:
    """Warmup then cosine decay, as in nanoGPT / the plan."""
    warmup = cfg["training"]["warmup_steps"]
    max_steps = cfg["training"]["max_steps"]
    base_lr = cfg["training"]["learning_rate"]
    min_lr = cfg["training"].get("min_lr", base_lr * 0.1)
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup) / max(max_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device, eval_iters: int) -> dict:
    """Mean train / val cross-entropy over `eval_iters` random batches."""
    model.eval()
    losses = {}
    for split, data in (("train", train_data), ("val", val_data)):
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(
                data, cfg["training"]["batch_size"], cfg["model"]["block_size"], device
            )
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                _, loss, _ = model(x, targets=y)
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint .pt to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps

    torch.manual_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    # --- Tokenizer: train once, reuse afterwards -------------------------
    tok_path = cfg["paths"]["tokenizer"]
    if os.path.exists(tok_path):
        tokenizer = load_tokenizer(tok_path)
        print(f"loaded tokenizer from {tok_path} (vocab {len(tokenizer.get_vocab())})")
    else:
        print("training BPE tokenizer ...")
        if cfg["data"]["dataset"] == "tinystories":
            texts = load_tinystories_texts("train", cfg["data"].get("max_train_stories"))
        else:
            texts = [download_tiny_shakespeare(cfg["data"]["tinyshakespeare_path"])]
        tokenizer = train_bpe_tokenizer(texts, cfg["model"]["vocab_size"], tok_path)
        print(f"trained tokenizer, saved to {tok_path}")

    # --- Data: encode once into flat arrays ------------------------------
    def encode_fn(text: str):
        return encode(tokenizer, text, add_bos=True, add_eos=True)

    if cfg["data"]["dataset"] == "tinystories":
        train_texts = load_tinystories_texts("train", cfg["data"].get("max_train_stories"))
        val_texts = load_tinystories_texts("validation", cfg["data"].get("max_val_stories", 2000))
    else:
        full = download_tiny_shakespeare(cfg["data"]["tinyshakespeare_path"])
        cut = int(len(full) * 0.9)
        train_texts, val_texts = [full[:cut]], [full[cut:]]
    print(f"encoding {len(train_texts)} train texts, {len(val_texts)} val texts ...")
    train_data = encode_texts_to_array(encode_fn, train_texts)
    val_data = encode_texts_to_array(encode_fn, val_texts)
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    # --- Model ------------------------------------------------------------
    model_cfg = TransformerConfig(
        vocab_size=cfg["model"]["vocab_size"],
        block_size=cfg["model"]["block_size"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=cfg["model"]["dropout"],
        pos_type=cfg["model"].get("pos_type", "learned"),
    )
    model = TransformerLM(model_cfg).to(device)
    print(f"model parameters: {model.num_parameters():,}")

    # AdamW with weight decay only on matrices (dim >= 2), as in nanoGPT.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["training"]["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["training"]["learning_rate"],
        betas=tuple(cfg["training"].get("betas", [0.9, 0.95])),
    )

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    model.train()
    log_csv = os.path.join(cfg["paths"]["log_dir"], cfg["paths"].get("log_name", "train_log.csv"))
    log_json = log_csv.replace(".csv", ".json")
    write_header = not os.path.exists(log_csv) or start_step == 0
    log_rows: list[dict] = []
    if os.path.exists(log_json) and start_step > 0:
        try:
            with open(log_json) as f:
                log_rows = json.load(f)
        except Exception:
            log_rows = []

    max_steps = cfg["training"]["max_steps"]
    eval_interval = cfg["training"]["eval_interval"]
    ckpt_interval = cfg["training"]["checkpoint_interval"]
    t0 = time.time()

    for step in range(start_step, max_steps):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(
            train_data, cfg["training"]["batch_size"], cfg["model"]["block_size"], device
        )
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            _, loss, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step:5d}/{max_steps}  loss {loss.item():.4f}  "
                  f"lr {lr:.2e}  elapsed {time.time() - t0:.0f}s")

        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(
                model, train_data, val_data, cfg, device, cfg["training"]["eval_iters"]
            )
            row = {
                "step": step,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "val_perplexity": round(math.exp(losses["val"]), 2),
                "lr": lr,
                "elapsed_s": round(time.time() - t0, 1),
            }
            log_rows.append(row)
            print(f"  eval step {step}: train_loss {row['train_loss']:.4f}  "
                  f"val_loss {row['val_loss']:.4f}  val_ppl {row['val_perplexity']:.2f}")
            with open(log_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(log_rows)
            with open(log_json, "w") as f:
                json.dump(log_rows, f, indent=2)

        if (step + 1) % ckpt_interval == 0 or step == max_steps - 1:
            ckpt_path = os.path.join(
                cfg["paths"]["checkpoint_dir"], f"ckpt_step_{step + 1}.pt"
            )
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": cfg,
                "train_loss": log_rows[-1]["train_loss"] if log_rows else None,
                "val_loss": log_rows[-1]["val_loss"] if log_rows else None,
            }, ckpt_path)
            print(f"  saved checkpoint {ckpt_path}")

    print(f"done. {max_steps - start_step} steps in {time.time() - t0:.0f}s. log: {log_csv}")


if __name__ == "__main__":
    main()
