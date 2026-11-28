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
| Baseline: 6L, 6H, learned, ctx 256 | 13,792,128 | 2.0967 | 2.1619 | 8.69 | 2,381 s |
| 1 head | 13,792,128 | 2.1257 | 2.1868 | 8.91 | 2,162 s |
| Sinusoidal pos | 13,693,824 | 3.0971 | 3.1244 | 22.75 | 2,360 s |
| 2 layers | 6,711,168 | 2.3038 | 2.3753 | 10.75 | 1,130 s |
| 4 layers | 10,251,648 | 2.1698 | 2.2258 | 9.26 | 1,750 s |
| Context 128 | 13,742,976 | 2.3064 | 2.3643 | 10.64 | 1,100 s |
| Context 512 | 13,890,432 | — | — | — | — |

## Status — 28 Nov 2026 (final)

6 of 7 training configs finished. Baseline and 1-head finished on Colab
(28 Nov re-runs; the 27 Nov baseline attempt was killed at step 20 by an
accidental interrupt and re-run). Sinusoidal, 2-layer, 4-layer and context-128
finished on Kaggle. Context-512 did not finish — its last verified state was
step 2,460/3,000 with a step-2,000 validation perplexity of 8.41 (mid-run,
not a final); its final number is unknown.

**Provenance warning.** The Kaggle session ended before the five Kaggle
configs' files were downloaded, so their log CSVs, checkpoints and generation
samples are lost — the notebook has no saved versions and Kaggle does not
retain a stopped session's working directory. The numbers in the table above
were read from the logs live, while the runs were still verifiable, and the
elapsed times come from the same logs. Nothing in the table is invented;
where data is missing, it is marked missing. Baseline and 1-head logs,
checkpoints and generation samples are safe on Google Drive.

- **Baseline — FINISHED 02:20 IST 28 Nov (Colab re-run).** Step 2999/3000,
  train loss 2.0967, val loss 2.1619, val perplexity 8.69, elapsed 2,380.8 s.
  Generation test: 4 samples, all coherent TinyStories-style, greedy
  cleanest.
- **1 head — FINISHED ~03:01 IST 28 Nov (Colab re-run).** Step 2999/3000,
  train loss 2.1257, val loss 2.1868, val perplexity 8.91, elapsed 2,162.2 s.
  Identical losses to the 27 Nov run. Generation test: 2 samples, coherent
  with minor small-model repetition.
- **Sinusoidal pos — FINISHED (Kaggle).** Step 2999/3000, train loss 3.0971,
  val loss 3.1244, val perplexity 22.75, elapsed 2,360.0 s. By far the worst
  config.
- **2 layers — FINISHED (Kaggle).** Step 2999/3000, train loss 2.3038, val
  loss 2.3753, val perplexity 10.75, elapsed 1,130.2 s.
- **4 layers — FINISHED (Kaggle).** Step 2999/3000, train loss 2.1698, val
  loss 2.2258, val perplexity 9.26, elapsed 1,750.0 s.
- **Context 128 — FINISHED (Kaggle).** Step 2999/3000, train loss 2.3064,
  val loss 2.3643, val perplexity 10.64, elapsed 1,099.5 s. One generation
  sample survives in the notebook ("Once upon a time there was a girl, who
  loved to play with her toys…").
- **Context 512 — NOT FINISHED.** No step-2999 line, no checkpoint, no
  generation file. Last verified state 06:49 IST 28 Nov: step 2,460/3,000.
- **Experiment 5 (temperature/top-k)** is done in practice per finished
  config (the generation tests above); the Kaggle configs' samples were
  lost with the session.
- **Parameter counts** in the table were verified by instantiating each
  config (sinusoidal has no learned position table, so it has 98,304 fewer
  parameters than the baseline; head count does not change the count).

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
