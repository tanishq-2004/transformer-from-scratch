"""Gradio demo for the Transformer from Scratch model.

Loads the final checkpoint (step 8000, trained on TinyStories) and generates
TinyStories-style text from a prompt. Runs on Hugging Face Spaces ZeroGPU.
"""

from __future__ import annotations

import gradio as gr
import spaces
import torch
import yaml

from src.model import TransformerConfig, TransformerLM
from src.tokenizer import decode, load_tokenizer

CHECKPOINT = "checkpoints/ckpt_step_8000.pt"
TOKENIZER = "checkpoints/tokenizer.json"
CONFIG = "configs/small.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = load_tokenizer(TOKENIZER)

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

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
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()


@spaces.GPU
def generate(prompt, max_new_tokens, temperature, top_k):
    bos = tokenizer.token_to_id("<bos>")
    prompt_ids = [bos] + tokenizer.encode(prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if int(top_k) > 0 else None,
        )
    return decode(tokenizer, out[0].tolist())


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", value="Once upon a time"),
        gr.Slider(10, 200, value=80, step=10, label="Max new tokens"),
        gr.Slider(0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)"),
        gr.Slider(0, 100, value=50, step=5, label="Top-k (0 = off)"),
    ],
    outputs=gr.Textbox(label="Generated story"),
    title="Transformer from Scratch — TinyStories Text Generation",
    description=(
        "A decoder-only GPT-style Transformer built in PyTorch from scratch "
        "(no nn.Transformer, no HF model). 6 layers, 6 heads, 384 hidden, "
        "~14M parameters, trained on a 200,000-story TinyStories subset for "
        "8,000 steps. Final: train loss 1.7893, val loss 1.8295, "
        "val perplexity 6.23."
    ),
    examples=[
        ["Once upon a time", 80, 0.8, 50],
        ["The little girl", 80, 0.8, 50],
        ["Tom was", 80, 0.8, 50],
    ],
)

if __name__ == "__main__":
    demo.launch()
