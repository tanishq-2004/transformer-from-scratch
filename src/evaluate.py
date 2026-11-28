"""Evaluation: perplexity on a validation or test set + generation samples.

Usage:
    python -m src.evaluate --checkpoint checkpoints/ckpt_step_5000.pt \
        --tokenizer checkpoints/tokenizer.json --config configs/small.yaml \
        --max_val_stories 2000

    python -m src.evaluate --checkpoint checkpoints/ckpt_step_8000.pt \
        --tokenizer checkpoints/tokenizer.json --config configs/small.yaml \
        --split test --output logs/eval_test.json

Reports, from the actual checkpoint (nothing invented):
  - cross-entropy loss and perplexity = exp(loss) on the chosen split
    (validation, or test — the test split is the second half of a seed-42
    shuffle of the validation stories, disjoint from the validation set.
    Note: TinyStories' official splits contain exact duplicate stories, so
    ~2.5% of the eval stories also appear in the train subset; pass
    --dedupe to replace them with fresh validation stories, keeping the
    set at full size and genuinely held out)
  - generation samples for a fixed set of prompts at temperature 0.8 / top-k 50
  - model size: parameter count, in-memory size (float32), checkpoint file size
  - memory: CPU peak RSS, GPU peak allocated (or unavailable if no GPU)
  - inference latency: median wall time for one greedy generation of
    max_new_tokens tokens, plus generation tokens/sec
Writes the results to logs/eval_<step>.json when --output is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import time

import numpy as np
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


def build_eval_texts(split, max_val_stories=2000, max_test_stories=2000,
                     dedupe=False, train_hashes_path=None):
    """Build the stories to evaluate perplexity on.

    Validation = the first max_val_stories of a seed-42 shuffle of the
    validation split; test = the next max_test_stories of the same shuffle.
    The two sets are disjoint by construction.

    With dedupe=True, stories whose exact text also appears in the train
    subset (TinyStories' official splits contain duplicates) are replaced
    with fresh validation stories from later in the same shuffle, so the
    set stays at full size and is genuinely held out.
    Returns (eval_texts, dropped_overlap, replaced).
    """
    pool = load_tinystories_texts(
        "validation", max_val_stories + max_test_stories + 2000)
    if split == "test":
        base = pool[max_val_stories:max_val_stories + max_test_stories]
        # Replacements come from a different range than the validation
        # set's, so the two deduped sets stay disjoint.
        candidates = pool[max_val_stories + max_test_stories + 1000:]
    else:
        base = pool[:max_val_stories]
        candidates = pool[max_val_stories + max_test_stories:
                          max_val_stories + max_test_stories + 1000]
    dropped = 0
    replaced = 0
    if dedupe:
        if train_hashes_path is None:
            train_hashes_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "data", "train_story_hashes.txt")
        if not os.path.exists(train_hashes_path):
            raise SystemExit(
                "dedupe needs data/train_story_hashes.txt — "
                "run `python build_train_hashes.py` first")
        with open(train_hashes_path) as f:
            train_hashes = {line.strip() for line in f}

        def is_clean(text):
            return hashlib.sha256(text.encode("utf-8")).hexdigest() not in train_hashes

        kept = [t for t in base if is_clean(t)]
        dropped = len(base) - len(kept)
        target = len(base)
        fill = [t for t in candidates if is_clean(t)][:target - len(kept)]
        replaced = len(fill)
        eval_texts = kept + fill
        if len(eval_texts) != target:
            raise SystemExit(
                f"dedupe could only build {len(eval_texts)} held-out stories "
                f"(wanted {target}) — the replacement pool ran out")
        print(f"dedupe: dropped {dropped} stories also in the train subset, "
              f"replaced with {replaced} fresh validation stories; "
              f"{len(eval_texts)} held-out stories")
    else:
        eval_texts = list(base)
    return eval_texts, dropped, replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_val_stories", type=int, default=2000)
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="which split to evaluate perplexity on")
    parser.add_argument("--max_test_stories", type=int, default=2000)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--latency_reps", type=int, default=5,
                        help="repetitions for the median inference-latency measurement")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed: same seed gives the same batches and samples")
    parser.add_argument("--dedupe", action="store_true",
                        help="drop eval stories whose exact text appears in the "
                             "train subset (TinyStories splits contain duplicates) "
                             "and replace them with fresh validation stories, "
                             "keeping the set at full size; "
                             "needs data/train_story_hashes.txt from "
                             "build_train_hashes.py")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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

    # Model size: parameter count, in-memory size (float32), checkpoint file size.
    num_params = model.num_parameters()
    param_bytes_mb = num_params * 4 / 1024 ** 2  # float32 parameters
    ckpt_file_mb = os.path.getsize(args.checkpoint) / 1024 ** 2
    print(f"model parameters: {num_params:,}")
    print(f"model size (params, float32): {param_bytes_mb:.1f} MB")
    print(f"checkpoint file: {ckpt_file_mb:.1f} MB (includes optimizer state)")

    eval_texts, dropped_overlap, replaced = build_eval_texts(
        args.split, args.max_val_stories, args.max_test_stories,
        dedupe=args.dedupe)

    eval_data = encode_texts_to_array(
        lambda t: encode(tokenizer, t, add_bos=True, add_eos=True), eval_texts
    )
    total = 0.0
    with torch.no_grad():
        for _ in range(args.eval_iters):
            x, y = get_batch(eval_data, cfg["training"]["batch_size"],
                             cfg["model"]["block_size"], device)
            _, loss, _ = model(x, targets=y)
            total += loss.item()
    split_loss = total / args.eval_iters
    perplexity = math.exp(split_loss)
    print(f"{args.split}_loss: {split_loss:.4f}")
    print(f"{args.split}_perplexity: {perplexity:.2f}  (on {len(eval_texts)} {args.split} stories, "
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

    # Inference latency: median wall time for one greedy generation of
    # max_new_tokens tokens, plus generation tokens/sec. Greedy (temp 0)
    # so the measurement is reproducible.
    bench_ids = torch.tensor(
        [[bos] + tokenizer.encode("Once upon a time").ids], dtype=torch.long, device=device
    )
    latencies = []
    with torch.no_grad():
        for _ in range(args.latency_reps):
            t0 = time.perf_counter()
            model.generate(bench_ids, max_new_tokens=args.max_new_tokens,
                           temperature=0.0, top_k=None)
            latencies.append(time.perf_counter() - t0)
    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    gen_tokens_per_sec = args.max_new_tokens / median_latency
    print(f"\ninference latency (median of {args.latency_reps}, "
          f"{args.max_new_tokens} tokens, {device}): {median_latency:.3f}s  "
          f"({gen_tokens_per_sec:.1f} tokens/s)")

    # Memory: CPU peak RSS for this process, GPU peak allocated if a GPU
    # is present (None = unavailable, per the checklist's marking rule).
    cpu_peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if torch.cuda.is_available():
        gpu_peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        gpu_total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    else:
        gpu_peak_mb = None
        gpu_total_mb = None
    print(f"\nCPU peak memory (RSS): {cpu_peak_rss_mb:.0f} MB")
    if gpu_peak_mb is not None:
        print(f"GPU peak allocated: {gpu_peak_mb:.0f} MB of {gpu_total_mb:.0f} MB")
    else:
        print("GPU memory: unavailable (no CUDA device)")

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        f"{args.split}_loss": round(split_loss, 4),
        f"{args.split}_perplexity": round(perplexity, 2),
        "eval_stories": len(eval_texts),
        "eval_iters": args.eval_iters,
        "deduped": bool(args.dedupe),
        "dropped_overlap": dropped_overlap,
        "replaced": replaced,
        "device": device,
        "inference_latency_s_median": round(median_latency, 3),
        "generation_tokens_per_sec": round(gen_tokens_per_sec, 1),
        "latency_reps": args.latency_reps,
        "max_new_tokens": args.max_new_tokens,
        "model_params": num_params,
        "model_size_mb": round(param_bytes_mb, 1),
        "checkpoint_file_mb": round(ckpt_file_mb, 1),
        "cpu_peak_rss_mb": round(cpu_peak_rss_mb, 0),
        "gpu_peak_allocated_mb": round(gpu_peak_mb, 0) if gpu_peak_mb is not None else None,
        "gpu_total_mb": round(gpu_total_mb, 0) if gpu_total_mb is not None else None,
        "samples": samples,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
