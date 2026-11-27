# Experiments

These ablations distinguish this project from a basic tutorial implementation.
All of them are run as **small controlled runs** first — 50,000-story subset,
~2–3k steps each, same seed (42), same learning rate, same batch size,
same tokenizer — changing **one variable at a time** against the baseline
(6 layers, 6 heads, learned positional embeddings, context 256).
The best config is then retrained on the full 200,000-story run.

All configs are evaluated on the **same fixed validation set** with the
**same tokenizer**. Report only numbers from the actual runs.

## 1. Single-head vs multi-head attention

- Configs: `n_head=1` vs `n_head=6`, same `n_embd=384`.
- Question: do multiple heads learn different relations, or is one head enough?
- Report: train loss, val loss, perplexity for each.

## 2. Learned vs sinusoidal positional embeddings

- Configs: `pos_type=learned` vs `pos_type=sinusoidal`.
- Question: does a trainable positional embedding fit TinyStories better
  than the fixed sine/cosine formula?
- Report: train loss, val loss, perplexity for each.

## 3. Different Transformer depths

- Configs: `n_layer=2` vs `n_layer=4` vs `n_layer=6`.
- Question: how does perplexity scale with depth, and where do returns diminish?
- Report: train loss, val loss, perplexity, parameter count, training time.

## 4. Different context lengths

- Configs: `block_size=128` vs `256` vs `512`.
- Question: TinyStories are short (~200 tokens), so 128 truncates stories
  and 512 wastes compute (attention is O(n^2)). Where is the sweet spot?
- Report: train loss, val loss, perplexity, steps/sec for each.

## 5. Temperature and top-k sampling (inference only, no retraining)

- Same prompt, same trained model. Grid: temperature 0.2 / 0.8 / 1.2
  × top-k 10 / 50 / 100.
- Question: how do temperature and top-k trade coherence against diversity?
- Report: the generated texts side by side for 2–3 fixed prompts.

## 6. Training loss vs validation loss

- Logged every 500 steps for every run (`logs/train_log.csv`).
- Question: is the model overfitting? Does val loss flatten while train
  loss keeps dropping?
- Report: loss curves (plot from the CSV) for the final model.

## 7. Perplexity across model configurations

Summary table — one row per configuration above. Filled only with numbers
from the actual runs.

| Config | Params | Train loss | Val loss | Val perplexity | Training time |
|---|---|---|---|---|---|
| Baseline: 6L, 6H, learned, ctx 256 | 13,792,128 | — | — | — | — |
| 1 head | 13,792,128 | 2.1257 | 2.1868 | 8.91 | — |
| Sinusoidal pos | 13,792,128 | — | — | — | — |
| 2 layers | — | — | — | — | — |
| 4 layers | — | — | — | — | — |
| Context 128 | 13,792,128 | — | — | — | — |
| Context 512 | 13,792,128 | — | — | — | — |

## Status — 27 Nov 2026

Runs execute sequentially on a Colab T4 GPU via `run_ablations.sh`
(nohup, 3,000 steps each on the 50k-story subset, seed 42).
Each config differs from baseline in exactly one field (verified by diff).

- **1 head — FINISHED 06:05 IST 27 Nov.** Step 2999/3000, train loss
  2.1257, val loss 2.1868, val perplexity 8.91. Log + final checkpoint
  (`ckpt_step_3000.pt`) copied to Google Drive. Generation test in Colab
  (2 samples, "Once upon a time", temp 0.8, top-k 50): story-like but not
  fully coherent — broken/repetitive in places, not gibberish.
  Training time (from the log's `elapsed_s`) to be filled from the Drive log
  at final verification.
- **Sinusoidal pos — IN PROGRESS at last check (06:05 IST 27 Nov).**
  Step 1820/3000, val perplexity 32.81 at the step-1500 eval (mid-run,
  still dropping). Log + checkpoint not yet copied to Drive.
- **Baseline — KILLED at step 20/3000 (~05:14 IST 27 Nov)** by an
  accidental Interrupt on the launch cell. Must be re-run to completion
  after the other six, or the comparisons are meaningless.
- **2 layers, 4 layers, context 128, context 512 — NOT STARTED.**
- **BLOCKED since 06:17 IST 27 Nov:** Colab refuses a GPU backend
  ("Cannot connect to GPU backend — usage limits"), runtime disconnected.
  No training can resume until the limit clears or Pay As You Go compute
  units are bought. Keep-alive checks paused at Tanishq's instruction;
  1-head's finished results are safe on Drive, sinusoidal's partial
  progress lives only in the Colab VM and is lost if the VM is recycled.
- **Experiment 5 (temperature/top-k)** is done in practice per finished
  config (the generation test above); the greedy vs high-temperature
  comparison from the baseline checkpoint still needs the baseline re-run.

## How to run an ablation

Configs live in `configs/ablation_*.yaml` (each = `small.yaml` with
`max_train_stories: 50000`, `max_steps: 3000`, its own checkpoint dir and
log). All seven run sequentially in the background with:

```bash
bash run_ablations.sh   # nohup; skips configs whose log already has step 2999
```

Or one config by hand:

```bash
cp configs/small.yaml configs/ablation_1head.yaml
# edit n_head: 6 -> 1, and max_train_stories: 200000 -> 50000, max_steps: 8000 -> 3000
python -m src.train --config configs/ablation_1head.yaml
```

## Interview notes (fill in after the runs)

- Why did you choose 6 layers / 6 heads / learned positions / context 256?
  Answer with the table above, not with "the tutorial said so".
- What surprised you? Report unexpected results honestly — e.g. if 1 head
  beat 6 heads on the small subset, say so and explain why (short training,
  small data), and that the final choice came from the full run.
- Never compare this model's perplexity to GPT or to large pretrained models.
  It is a ~14M-parameter model trained from scratch on TinyStories.
