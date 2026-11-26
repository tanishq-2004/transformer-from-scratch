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

Summary table — one row per configuration above. Fill in only after the runs.

| Config | Params | Train loss | Val loss | Val perplexity | Training time |
|---|---|---|---|---|---|
| Baseline: 6L, 6H, learned, ctx 256 | _ | _ | _ | _ | _ |
| 1 head | _ | _ | _ | _ | _ |
| Sinusoidal pos | _ | _ | _ | _ | _ |
| 2 layers | _ | _ | _ | _ | _ |
| 4 layers | _ | _ | _ | _ | _ |
| Context 128 | _ | _ | _ | _ | _ |
| Context 512 | _ | _ | _ | _ | _ |

## How to run an ablation

Copy a config, change one field, run training:

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
