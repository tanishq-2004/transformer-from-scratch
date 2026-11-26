# Transformer from Scratch

A decoder-only GPT-style Transformer built in PyTorch **from scratch** — no
`nn.Transformer`, no Hugging Face model, no Trainer. Trained on TinyStories
for language modelling and autoregressive text generation.

## What is implemented from scratch

- BPE tokenization (trained on TinyStories, vocab 8000)
- Token embeddings + positional embeddings (learned or sinusoidal)
- Scaled dot-product attention, coded by hand
- Multi-head self-attention (ablation: 1 head vs 6 heads)
- Causal masking
- Feed-forward networks, residual connections, pre-LayerNorm
- Decoder-only Transformer stack (ablation: 2 / 4 / 6 layers)
- Cross-entropy loss and backpropagation (PyTorch autograd)
- Training on a real text corpus (TinyStories)
- Autoregressive text generation with KV cache, greedy + temperature / top-k

## Model (planned full config)

- 6 layers, 6 heads, 384 hidden, context 256, dropout 0.1, vocab 8000
- ~14M parameters (see `configs/small.yaml`)

## Dataset

Hugging Face `roneneldan/TinyStories` — 2,119,719 train / 21,990 validation
stories, synthetic English short stories. First full run: a 200,000-story
subset of the train split, to fit a Kaggle 12-hour session. Tiny Shakespeare
(~1 MB) is used only for the CPU smoke test.

## How to train

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test (CPU, ~a minute): 2 layers, Tiny Shakespeare, 50 steps
python -m src.train --config configs/smoke.yaml

# Full run (Kaggle T4 x2 — see notebooks/kaggle_train.ipynb)
python -m src.train --config configs/small.yaml
# Resume from a checkpoint:
python -m src.train --config configs/small.yaml --resume checkpoints/ckpt_step_1000.pt
```

Training logs train/val loss and perplexity to `logs/train_log.csv` and
`logs/train_log.json`. Checkpoints go to `checkpoints/ckpt_step_N.pt`.

## How to generate

```bash
python -m src.generate --checkpoint checkpoints/ckpt_step_8000.pt \
  --tokenizer checkpoints/tokenizer.json \
  --prompt "Once upon a time" --max_new_tokens 50 --temperature 0.8 --top_k 50
```

## How to evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/ckpt_step_8000.pt \
  --tokenizer checkpoints/tokenizer.json --config configs/small.yaml
```

## How to serve (CPU)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# GET /health, POST /generate {"prompt": "...", "max_new_tokens": 50}
docker build -t transformer-from-scratch .
docker run -p 8000:8000 transformer-from-scratch
```

The permanent resume link is the live demo below (Gradio on Hugging Face
Spaces ZeroGPU, free). Kaggle is for training only — its sessions end after
12 hours, so a Kaggle link is not a resume link.

**Live demo:** https://huggingface.co/spaces/Skywalker67/transformer-from-scratch

## Experiments

See `experiments.md`. Ablations are run as small controlled runs on a
50k-story subset (same seed, LR, batch; change one variable at a time),
then the best config is retrained on the full 200k-story run.

## Real metrics

All numbers below are from real runs — nothing here is an estimate or a guess.

- Smoke test (CPU, Tiny Shakespeare, 2-layer, 538,240 params, 50 steps, 10s):
  step 0 train 6.9270, val 6.9208, val perplexity 1013.14; step 49 train
  5.6711, val 5.6322, val perplexity 279.28. See `logs/smoke_test_log.txt`.
  Unit checks 10/10 passed.
- Full run (Colab T4 GPU, 200,000-story TinyStories subset, 8,000 steps):
  - train loss: **1.7893** (step 7999 eval)
  - val loss: **1.8295** (step 7999 eval)
  - val perplexity: **6.23** (step 7999 eval)
  - training time: 6,071 s (~101 min), ~1.42 steps/sec steady
  - start of run (step 0): train 9.0384, val 9.0383, val perplexity 8419.33
  - model: 6 layers, 6 heads, 384 hidden, context 256, 13,792,128 params
  - checkpoints: `ckpt_step_1000.pt`–`ckpt_step_8000.pt` (160 MB each) +
    `tokenizer.json`, backed up in Google Drive
  - generation sample (final checkpoint, "Once upon a time"): "there was a
    little girl named Lily. She loved to play with her toys all day long.
    One day, her mom surprised her with the big, pretty dress..."
- Ablations: see `experiments.md` (filled after the ablation runs)
