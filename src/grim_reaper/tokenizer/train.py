"""Train and load the real tokenizer: Hugging Face `tokenizers` byte-level BPE.

This is the same algorithm as `bpe.py` (see docs/bpe-and-data.md) implemented in fast Rust.
It trains on the *train split only*, so validation text never shapes the vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ..config import ConfigError, RunConfig
from ..data.stories import read_stories, sha256_file, split_stories
from .bpe import SPECIAL_TOKEN

TOKENIZER_TRAIN_STORIES = 50_000


def sidecar_path(tokenizer_path: str | Path) -> Path:
    """`tokenizer.json` -> `tokenizer.meta.json`: records the split it was built for."""
    return Path(tokenizer_path).with_suffix(".meta.json")


def split_identity(cfg: RunConfig, source_sha256: str) -> dict:
    """Everything that decides which stories the tokenizer saw. `prepare` checks it matches."""
    return {
        "source_sha256": source_sha256,
        "seed": cfg.train.seed,
        "max_stories": cfg.data.max_stories,
        "val_fraction": cfg.data.val_fraction,
        "tokenizer_train_stories": TOKENIZER_TRAIN_STORIES,
        "vocab_size": cfg.model.vocab_size,
    }


def train_tokenizer(cfg: RunConfig, stories_path: str | Path) -> Path:
    """Train on the train split of `stories_path`; write to cfg.data.tokenizer_path."""
    if cfg.model.vocab_size < 257:
        raise ConfigError(
            f"model.vocab_size must be at least 257 (256 bytes + end-of-text), "
            f"got {cfg.model.vocab_size}"
        )
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

    got = tokenizer.get_vocab_size()
    if got != cfg.model.vocab_size:
        raise ConfigError(
            f"corpus is too small for a {cfg.model.vocab_size}-token vocab (trained {got}); "
            f"raise data.max_stories or lower model.vocab_size"
        )

    out = Path(cfg.data.tokenizer_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    side = sidecar_path(out)
    side.unlink(missing_ok=True)  # a crash between the two writes must not leave a stale pair
    tokenizer.save(str(out))
    side.write_text(json.dumps(split_identity(cfg, sha256_file(stories_path)), indent=2))
    return out


def load_tokenizer(path: str | Path) -> Tokenizer:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"tokenizer not found at {p}; run `grim-reaper tokenizer train`")
    try:
        return Tokenizer.from_file(str(p))
    except Exception as e:
        raise ConfigError(
            f"tokenizer at {p} is unreadable ({e}); rerun `grim-reaper tokenizer train`"
        ) from e
