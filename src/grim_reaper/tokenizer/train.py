"""Train and load the real tokenizer: Hugging Face `tokenizers` byte-level BPE.

This is the same algorithm as `bpe.py` (see docs/bpe-and-data.md) implemented in fast Rust.
It trains on the *train split only*, so validation text never shapes the vocabulary.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ..config import ConfigError, RunConfig
from ..data.stories import read_stories, split_stories
from .bpe import SPECIAL_TOKEN

TOKENIZER_TRAIN_STORIES = 50_000


def train_tokenizer(cfg: RunConfig, stories_path: str | Path) -> Path:
    """Train on the train split of `stories_path`; write to cfg.data.tokenizer_path."""
    stories, _ = read_stories(stories_path, cfg.data.max_stories)
    train, _ = split_stories(stories, cfg.data.val_fraction, cfg.train.seed)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.model.vocab_size,
        special_tokens=[SPECIAL_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(train[:TOKENIZER_TRAIN_STORIES], trainer)

    out = Path(cfg.data.tokenizer_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    return out


def load_tokenizer(path: str | Path) -> Tokenizer:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"tokenizer not found at {p}; run `grim-reaper tokenizer train`")
    return Tokenizer.from_file(str(p))
