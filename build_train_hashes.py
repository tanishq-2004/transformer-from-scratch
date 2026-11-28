"""Build data/train_story_hashes.txt: one SHA-256 per story in the
200,000-story train subset — the exact texts the model and the BPE
tokenizer trained on.

TinyStories' official train and validation splits contain exact duplicate
stories (the corpus was generated from templated prompts), so ~2.5% of
the validation/test stories also appear in training. src/evaluate.py
--dedupe reads this file and drops those stories, so the evaluated set is
genuinely held out.

The file is local-only (data/ is gitignored); regenerate with:
    python build_train_hashes.py
"""

from __future__ import annotations

import hashlib
import os

from src.data import load_tinystories_texts


def main():
    texts = load_tinystories_texts("train", 200_000)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "train_story_hashes.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for text in texts:
            f.write(hashlib.sha256(text.encode("utf-8")).hexdigest() + "\n")
    print(f"wrote {len(texts)} train-story hashes to {out_path}")


if __name__ == "__main__":
    main()
