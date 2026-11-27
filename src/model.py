"""Decoder-only Transformer implemented from scratch in PyTorch.

No nn.Transformer, no Hugging Face model, no Trainer.
Every component is written here so it can be explained in an interview:
  - token embeddings + positional embeddings (learned or sinusoidal)
  - scaled dot-product attention, coded by hand
  - multi-head self-attention
  - causal masking
  - feed-forward network
  - residual connections + pre-LayerNorm
  - cross-entropy loss is applied in train.py; backprop is PyTorch autograd
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """All model hyper-parameters. Loaded from configs/*.yaml."""

    vocab_size: int = 8000
    block_size: int = 256          # context length
    n_layer: int = 6               # depth — ablation: 2 / 4 / 6
    n_head: int = 6                # ablation: 1 (single-head) vs 6 (multi-head)
    n_embd: int = 384              # hidden size; head_dim = n_embd // n_head
    dropout: float = 0.1
    pos_type: str = "learned"      # ablation: "learned" vs "sinusoidal"
    bias: bool = False             # no bias in Linear / LayerNorm, as in nanoGPT


def sinusoidal_positional_embedding(block_size: int, n_embd: int) -> torch.Tensor:
    """Fixed sinusoidal positional encodings, as in 'Attention Is All You Need'.

    Returns a (block_size, n_embd) tensor. Not learned; added to token
    embeddings exactly like the learned version, so the two are swappable.
    """
    position = torch.arange(block_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, n_embd, 2, dtype=torch.float)
        * (-math.log(10000.0) / n_embd)
    )
    pe = torch.zeros(block_size, n_embd)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CausalSelfAttention(nn.Module):
    """One self-attention head-group, with causal masking, coded by hand."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Q, K, V projections in a single Linear (faster, same maths).
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Causal mask: (1, 1, block_size, block_size), 1 = keep, 0 = mask out.
        # Registered as a buffer so it moves with .to(device) but is not a parameter.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None, use_cache: bool = False):
        """x: (B, T, C). past_kv: optional (past_k, past_v) for KV-cache decoding.

        Returns (y, present_kv). present_kv is None unless use_cache is True.
        """
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        total_len = k.size(2)  # past length + current length

        # --- Scaled dot-product attention, by hand ---
        # scores: (B, n_head, T, total_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask: a query at position i may only attend to keys j <= i.
        # When generating one token at a time, past_len offsets the positions.
        past_len = total_len - T
        mask = self.bias[:, :, past_len:past_len + T, :total_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        # Weighted sum of values: (B, n_head, T, head_dim)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv


class MLP(nn.Module):
    """Position-wise feed-forward network: C -> 4C (GELU) -> C."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One Transformer block: pre-LayerNorm + attention + MLP, with residuals."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache: bool = False):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out  # residual connection 1
        x = x + self.mlp(self.ln_2(x))  # residual connection 2
        return x, present_kv


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model (GPT-style)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_type == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
            self.register_buffer("sinusoidal_pe", None)
        elif config.pos_type == "sinusoidal":
            self.position_embedding = None
            self.register_buffer(
                "sinusoidal_pe",
                sinusoidal_positional_embedding(config.block_size, config.n_embd),
            )
        else:
            raise ValueError(f"unknown pos_type: {config.pos_type!r} (use 'learned' or 'sinusoidal')")
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        # Output head. Tied to the token embedding to save parameters.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Small init for the output / residual projections, as in nanoGPT.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None, use_cache: bool = False):
        """idx: (B, T) token ids. targets: (B, T) next-token ids or None.

        Returns (logits, loss, present_key_values).
        Loss is cross-entropy on next-token prediction when targets are given.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        # With a KV cache, the new tokens continue after the cached ones,
        # so their positions must be offset by the cached length — otherwise
        # every generated token would get position 0 and the cache would
        # silently produce different outputs from a full forward pass.
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].size(2)
        positions = torch.arange(past_len, past_len + T, device=idx.device)
        tok_emb = self.token_embedding(idx)
        if self.config.pos_type == "learned":
            pos_emb = self.position_embedding(positions)
        else:
            pos_emb = self.sinusoidal_pe[positions]
        x = self.drop(tok_emb + pos_emb)

        present_key_values = []
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_kv=past, use_cache=use_cache)
            present_key_values.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy over the vocab, flattened over batch * time.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, (present_key_values if use_cache else None)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 use_cache: bool = True):
        """Autoregressive generation: feed each new token back in.

        idx: (B, T) prompt. Greedy when temperature == 0.
        Top-p (nucleus) sampling keeps the smallest token set whose
        cumulative probability reaches top_p.
        Returns (B, T + max_new_tokens) token ids.
        """
        self.eval()
        past_key_values = None
        for _ in range(max_new_tokens):
            # Only the last token needs a forward pass once the cache is warm,
            # but the first step needs the full prompt (cropped to block_size).
            idx_cond = idx if past_key_values is None else idx[:, -1:]
            if idx_cond.size(1) > self.config.block_size:
                idx_cond = idx_cond[:, -self.config.block_size:]
                past_key_values = None  # cache no longer aligns after cropping
            logits, _, past_key_values = self(
                idx_cond, past_key_values=past_key_values, use_cache=use_cache
            )
            logits = logits[:, -1, :]  # next-token logits for the last position
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    # Keep only the top-k logits, mask the rest with -inf.
                    top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(
                        logits < top_values[:, [-1]], float("-inf")
                    )
                if top_p is not None:
                    # Nucleus sampling: keep the smallest set of tokens
                    # whose cumulative probability reaches top_p.
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_to_remove = cumulative_probs - sorted_probs > top_p
                    sorted_logits[sorted_to_remove] = float("-inf")
                    logits = torch.zeros_like(logits).scatter(
                        -1, sorted_indices, sorted_logits
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
