"""Data loading: TinyStories (real corpus) and Tiny Shakespeare (smoke test).

- TinyStories: Hugging Face `roneneldan/TinyStories`.
  Full split: 2,119,719 train / 21,990 validation stories.
  The first full run uses a 200,000-story subset of the train split so it
  fits a Kaggle 12-hour session.
- Tiny Shakespeare: ~1 MB of text, downloaded from Karpathy's repo.
  Used only for the tiny CPU smoke test.

Text is tokenized once into a flat int64 array (train.bin / val.bin style,
kept in memory as a numpy array here), then batches are random
(block_size + 1)-token windows: x = tokens[i:i+block_size],
y = tokens[i+1:i+block_size+1]. This is the standard GPT data pipeline.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(path: str) -> str:
    """Download Tiny Shakespeare text (~1 MB) if not already present."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_tinystories_texts(split: str = "train", max_stories: int | None = None) -> list[str]:
    """Load TinyStories story texts from Hugging Face.

    Requires the `datasets` package. Returns a list of story strings.
    `max_stories` caps the split (e.g. 200_000 for the first full run,
    50_000 for ablations). The dataset is shuffled with a fixed seed
    before capping so the subset is representative.
    """
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split)
    if max_stories is not None and len(ds) > max_stories:
        ds = ds.shuffle(seed=42).select(range(max_stories))
    return ds["text"]


def encode_texts_to_array(encode_fn, texts: list[str]) -> np.ndarray:
    """Encode a list of texts into one flat int64 array, stories joined by <eos>.

    encode_fn maps one story string to a list of token ids (including <bos>/<eos>).
    """
    all_ids: list[int] = []
    for text in texts:
        all_ids.extend(encode_fn(text))
    return np.array(all_ids, dtype=np.int64)


def get_batch(data: np.ndarray, batch_size: int, block_size: int,
              device: str, generator: torch.Generator | None = None):
    """Sample a random batch of (x, y) next-token pairs from a flat token array."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)
