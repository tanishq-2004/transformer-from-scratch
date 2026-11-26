"""FastAPI serving for the trained Transformer.

Endpoints:
  GET  /health    -> {"status": "ok", "model": ..., "device": "cpu"}
  POST /generate  -> {"prompt": ..., "generated": ..., "tokens": ...}
                     body: {"prompt": str, "max_new_tokens": 50,
                            "temperature": 0.8, "top_k": 50}

Loads the checkpoint on CPU at startup. Model weights and tokenizer are
expected at the paths given by MODEL_PATH / TOKENIZER_PATH (defaults below).
Run locally: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from src.model import TransformerConfig, TransformerLM
from src.tokenizer import decode, load_tokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/ckpt_step_8000.pt")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "checkpoints/tokenizer.json")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/small.yaml")

app = FastAPI(title="Transformer from Scratch — text generation")

_model = None
_tokenizer = None


class GenerateRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 50


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"checkpoint not found at {MODEL_PATH}. Train first, or set MODEL_PATH."
        )
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    _tokenizer = load_tokenizer(TOKENIZER_PATH)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model_cfg = TransformerConfig(
        vocab_size=cfg["model"]["vocab_size"],
        block_size=cfg["model"]["block_size"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=0.0,
        pos_type=cfg["model"].get("pos_type", "learned"),
    )
    _model = TransformerLM(model_cfg)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    return _model, _tokenizer


@app.on_event("startup")
def startup():
    # Load lazily only if files exist, so /health works even before training.
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        _load_model()


@app.get("/health")
def health():
    loaded = _model is not None
    return {"status": "ok", "model_loaded": loaded, "device": "cpu"}


@app.post("/generate")
def generate(req: GenerateRequest):
    model, tokenizer = _load_model()
    bos = tokenizer.token_to_id("<bos>")
    prompt_ids = [bos] + tokenizer.encode(req.prompt).ids
    # Crop the prompt to the model's context length.
    with open(CONFIG_PATH) as f:
        block_size = yaml.safe_load(f)["model"]["block_size"]
    prompt_ids = prompt_ids[-block_size:]
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k if req.top_k > 0 else None,
        )
    text = decode(tokenizer, out[0].tolist())
    return {"prompt": req.prompt, "generated": text, "tokens": len(out[0])}
