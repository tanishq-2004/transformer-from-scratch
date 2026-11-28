"""Verify the 2,000-story validation and test sets are held out.

Builds both sets with the same code path the evaluator uses
(src/evaluate.py build_eval_texts, with --dedupe replacement) and checks,
empirically:

1. Each set has exactly 2,000 stories.
2. The two sets are disjoint from each other.
3. No story in either set appears in the 200,000-story train subset the
   model and the tokenizer were trained on.
4. Every story in either set comes from the TinyStories validation split.
5. No duplicates within either set.

The tokenizer and the model were trained only on the train split — see
src/train.py (tokenizer: load_tinystories_texts("train", ...) at line ~110;
model training data: load_tinystories_texts("train", ...) at line ~121).
Combined with check 3, that means no eval story was used for tokenizer
training or model training.

Usage:
    python verify_test_split.py
"""

from __future__ import annotations

import hashlib

from src.data import load_tinystories_texts
from src.evaluate import build_eval_texts


def story_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    print("building deduped validation and test sets (same code as the evaluator)...")
    val_texts, val_dropped, val_replaced = build_eval_texts(
        "validation", dedupe=True)
    test_texts, test_dropped, test_replaced = build_eval_texts(
        "test", dedupe=True)
    print(f"validation: dropped {val_dropped}, replaced {val_replaced}")
    print(f"test: dropped {test_dropped}, replaced {test_replaced}")

    print("loading train subset (200,000 stories, seed-42 shuffle)...")
    train_texts = load_tinystories_texts("train", 200_000)
    print("loading full validation split (21,990 stories)...")
    val_full = load_tinystories_texts("validation", None)

    train_hashes = {story_hash(t) for t in train_texts}
    val_hashes = {story_hash(t) for t in val_texts}
    test_hashes = {story_hash(t) for t in test_texts}
    val_full_hashes = {story_hash(t) for t in val_full}

    checks = []
    checks.append(("validation set has exactly 2,000 stories",
                   len(val_texts) == 2000, f"got {len(val_texts)}"))
    checks.append(("test set has exactly 2,000 stories",
                   len(test_texts) == 2000, f"got {len(test_texts)}"))
    checks.append(("validation and test sets are disjoint",
                   len(val_hashes & test_hashes) == 0,
                   f"overlap {len(val_hashes & test_hashes)}"))
    checks.append(("no validation story appears in the train subset",
                   len(val_hashes & train_hashes) == 0,
                   f"overlap {len(val_hashes & train_hashes)}"))
    checks.append(("no test story appears in the train subset",
                   len(test_hashes & train_hashes) == 0,
                   f"overlap {len(test_hashes & train_hashes)}"))
    checks.append(("every validation story comes from the validation split",
                   val_hashes <= val_full_hashes,
                   f"{len(val_hashes - val_full_hashes)} outside validation"))
    checks.append(("every test story comes from the validation split",
                   test_hashes <= val_full_hashes,
                   f"{len(test_hashes - val_full_hashes)} outside validation"))
    checks.append(("no duplicates within the validation set",
                   len(val_hashes) == len(val_texts),
                   f"{len(val_texts)} stories, {len(val_hashes)} unique"))
    checks.append(("no duplicates within the test set",
                   len(test_hashes) == len(test_texts),
                   f"{len(test_texts)} stories, {len(test_hashes)} unique"))

    all_pass = True
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name} ({detail})")
        all_pass = all_pass and passed
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
