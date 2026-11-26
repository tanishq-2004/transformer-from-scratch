"""Autoregressive text generation from a trained checkpoint.

Usage:
    python -m src.generate --checkpoint checkpoints/ckpt_step_5000.pt \
        --tokenizer checkpoints/tokenizer.json --prompt "Once upon a time" \
        --max_new_tokens 50 --temperature 0.8 --top_k 50

The model generates one token at a time and feeds it back in (autoregressive),
using the KV cache in model.py so past tokens are not recomputed.
Greedy decoding when --temperature 0.
"""

from __future__ import annotations

import argparse

import torch
import yaml

from src.model import TransformerConfig, TransformerLM
from src.tokenizer import decode, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
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

    bos = tokenizer.token_to_id("<bos>")
    prompt_ids = [bos] + tokenizer.encode(args.prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )
    print(decode(tokenizer, out[0].tolist()))


if __name__ == "__main__":
    main()
