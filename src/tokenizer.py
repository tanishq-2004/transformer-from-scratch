"""BPE tokenization.

Trains a byte-level BPE tokenizer (Hugging Face `tokenizers` library) on a
text corpus, saves it as tokenizer.json, and provides encode / decode helpers
used identically by training, evaluation and serving.
"""

from __future__ import annotations

import json
import os

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Special tokens. Their ids are fixed after training; <pad> is id 0 so it
# can be ignored by the loss if padding is ever used.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe_tokenizer(texts, vocab_size: int, save_path: str) -> Tokenizer:
    """Train a byte-level BPE tokenizer on an iterable of texts.

    Args:
        texts: iterable of str (stories / documents).
        vocab_size: total vocabulary size including special tokens.
        save_path: where to write tokenizer.json.

    Returns the trained Tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    # Without a ByteLevel decoder, decode() leaves "Ġ" space-markers in the
    # text (e.g. "Ġupon" instead of " upon"). The decoder reverses them.
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a tokenizer previously saved with train_bpe_tokenizer()."""
    return Tokenizer.from_file(path)


def encode(tokenizer: Tokenizer, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    """Encode text to token ids, optionally wrapped with <bos> ... <eos>."""
    ids = tokenizer.encode(text).ids
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if add_bos:
        ids = [bos] + ids
    if add_eos:
        ids = ids + [eos]
    return ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    """Decode token ids back to text, skipping special tokens."""
    return tokenizer.decode(ids, skip_special_tokens=True)


def vocab_size_of(path: str) -> int:
    """Vocabulary size of a saved tokenizer.json (for config checks)."""
    with open(path) as f:
        data = json.load(f)
    return len(data["model"]["vocab"])
