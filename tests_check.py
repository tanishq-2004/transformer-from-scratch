"""Quick unit checks for the Transformer from Scratch code.

Run: python tests_check.py
Checks (all must pass):
  1. Causal attention mask: a query at position i cannot attend to keys j > i.
     Verified by feeding a one-hot sequence and checking that changing a
     future token does not change earlier positions' outputs.
  2. Model output shape: logits are (B, T, vocab_size).
  3. Tokenizer encode/decode round-trip: decode(encode(text)) == text
     (for plain ASCII text, modulo BPE spacing).

Writes results to logs/unit_check.txt and prints them.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.model import TransformerConfig, TransformerLM
from src.tokenizer import decode, encode, load_tokenizer

RESULTS = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=1000, block_size=32, n_layer=2,
                            n_head=2, n_embd=64, dropout=0.0, pos_type="learned")
    model = TransformerLM(cfg)
    model.eval()

    # 1. Output shape
    x = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        logits, loss, _ = model(x, targets=x)
    check("output shape (B,T,V)",
          tuple(logits.shape) == (2, 16, 1000),
          f"got {tuple(logits.shape)}")
    check("loss is finite and positive",
          loss is not None and torch.isfinite(loss).item() and loss.item() > 0,
          f"loss={loss.item():.4f}")

    # 2. Causal mask: changing token at position 10 must not change outputs
    #    at positions 0..9, but may change position 10 onward.
    with torch.no_grad():
        base = torch.randint(0, 1000, (1, 16))
        logits_base, _, _ = model(base)
        changed = base.clone()
        changed[0, 10] = (changed[0, 10] + 1) % 1000
        logits_changed, _, _ = model(changed)
    early_same = torch.allclose(logits_base[0, :10], logits_changed[0, :10], atol=1e-5)
    later_diff = not torch.allclose(logits_base[0, 10:], logits_changed[0, 10:], atol=1e-5)
    check("causal mask: earlier positions unchanged by future token",
          early_same, f"max diff early={ (logits_base[0,:10]-logits_changed[0,:10]).abs().max().item():.2e}")
    check("causal mask: later positions do change (sanity)",
          later_diff, "")

    # 3. Sinusoidal positional embeddings also work
    cfg_sin = TransformerConfig(vocab_size=1000, block_size=32, n_layer=2,
                                n_head=2, n_embd=64, dropout=0.0, pos_type="sinusoidal")
    model_sin = TransformerLM(cfg_sin)
    model_sin.eval()
    with torch.no_grad():
        logits_sin, _, _ = model_sin(x)
    check("sinusoidal positional embeddings run",
          tuple(logits_sin.shape) == (2, 16, 1000), "")

    # 4. Single-head attention runs (ablation config)
    cfg_1h = TransformerConfig(vocab_size=1000, block_size=32, n_layer=2,
                               n_head=1, n_embd=64, dropout=0.0, pos_type="learned")
    model_1h = TransformerLM(cfg_1h)
    model_1h.eval()
    with torch.no_grad():
        logits_1h, _, _ = model_1h(x)
    check("single-head (n_head=1) runs",
          tuple(logits_1h.shape) == (2, 16, 1000), "")

    # 5. Tokenizer round-trip (uses the smoke tokenizer trained by the smoke test)
    tok_path = "checkpoints/smoke_tokenizer.json"
    if os.path.exists(tok_path):
        tok = load_tokenizer(tok_path)
        text = "Once upon a time there was a little cat."
        ids = encode(tok, text, add_bos=False, add_eos=False)
        back = decode(tok, ids)
        check("tokenizer encode/decode round-trip",
              back.strip() == text.strip(), f"decoded={back!r}")
        check("tokenizer vocab size matches smoke config",
              len(tok.get_vocab()) == 1000, f"got {len(tok.get_vocab())}")
    else:
        print(f"[SKIP] tokenizer round-trip — {tok_path} not found (run smoke test first)")

    # 6. Generation produces the requested number of tokens
    with torch.no_grad():
        prompt = torch.randint(0, 1000, (1, 5))
        out = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    check("generation length = prompt + max_new_tokens",
          tuple(out.shape) == (1, 15), f"got {tuple(out.shape)}")

    # 7. KV-cache generation matches non-cached generation (greedy)
    with torch.no_grad():
        prompt = torch.randint(0, 1000, (1, 5))
        out_cached = model.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=True)
        out_nocache = model.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=False)
    check("KV-cache generation matches non-cached (greedy)",
          torch.equal(out_cached, out_nocache), "")

    os.makedirs("logs", exist_ok=True)
    with open("logs/unit_check.txt", "w") as f:
        for name, ok, detail in RESULTS:
            f.write(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}\n")
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
