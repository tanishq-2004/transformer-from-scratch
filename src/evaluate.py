"""Evaluation: perplexity on a validation set + generation samples.

Usage:
    python -m src.evaluate --checkpoint checkpoints/ckpt_step_5000.pt \
        --tokenizer checkpoints/tokenizer.json --config configs/small.yaml \
        --max_val_stories 2000

Reports, from the actual checkpoint (nothing invented):
  - validation cross-entropy loss
  - validation perplexity = exp(val loss)
  - generation samples for a fixed set of prompts at temperature 0.8 / top-k 50
Writes the results to logs/eval_<step>.json when --output is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import yaml

from src.data import encode_texts_to_array, get_batch, load_tinystories_texts
from src.model import TransformerConfig, TransformerLM
from src.tokenizer import decode, encode, load_tokenizer

DEFAULT_PROMPTS = [
    "Once upon a time",
    "Tom was a little boy",
    "The cat sat on the",
    "One sunny morning, Lily",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_val_stories", type=int, default=2000)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_cfg = TransformerConfig(
        vocab_size=cfg["model"]["vocab_size"],
        block_size=cfg["model"]["block_size"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=0.0,
        pos_type=cfg["model"].get("pos_type", "learned"),
    )
    model = TransformerLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_texts = load_tinystories_texts("validation", args.max_val_stories)
    val_data = encode_texts_to_array(
        lambda t: encode(tokenizer, t, add_bos=True, add_eos=True), val_texts
    )
    total = 0.0
    with torch.no_grad():
        for _ in range(args.eval_iters):
            x, y = get_batch(val_data, cfg["training"]["batch_size"],
                             cfg["model"]["block_size"], device)
            _, loss, _ = model(x, targets=y)
            total += loss.item()
    val_loss = total / args.eval_iters
    perplexity = math.exp(val_loss)
    print(f"val_loss: {val_loss:.4f}")
    print(f"val_perplexity: {perplexity:.2f}  (on {args.max_val_stories} val stories, "
          f"{args.eval_iters} batches)")

    samples = []
    bos = tokenizer.token_to_id("<bos>")
    with torch.no_grad():
        for prompt in DEFAULT_PROMPTS:
            idx = torch.tensor(
                [[bos] + tokenizer.encode(prompt).ids], dtype=torch.long, device=device
            )
            out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                                 temperature=0.8, top_k=50)
            text = decode(tokenizer, out[0].tolist())
            samples.append({"prompt": prompt, "generated": text})
            print(f"\nPROMPT: {prompt}\n{text}")

    result = {
        "checkpoint": args.checkpoint,
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(perplexity, 2),
        "max_val_stories": args.max_val_stories,
        "eval_iters": args.eval_iters,
        "samples": samples,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
