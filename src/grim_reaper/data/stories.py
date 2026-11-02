"""Read raw story text and split it into train and validation sets.

Both `tokenizer train` and `data prepare` go through `split_stories`, so they always agree on
which stories are validation. That matters: validation text must never influence the tokenizer.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..config import ConfigError, RunConfig
from ..tokenizer.bpe import SPECIAL_TOKEN


def read_stories(
    path: str | Path, max_stories: int | None = None, chunk_chars: int = 1 << 20
) -> tuple[list[str], int]:
    """Return (stories, dropped_empty). Stories in the file are separated by <|endoftext|>.

    The file is read in chunks so a multi-gigabyte dataset never sits in memory as one string;
    a separator that straddles a chunk boundary is handled by carrying the tail over.
    """
    stories: list[str] = []
    dropped = 0
    carry = ""

    def take(text: str) -> bool:
        nonlocal dropped
        text = text.strip()
        if not text:
            dropped += 1
            return False
        stories.append(text)
        return max_stories is not None and len(stories) >= max_stories

    with open(path, encoding="utf-8") as f:
        while chunk := f.read(chunk_chars):
            *done, carry = (carry + chunk).split(SPECIAL_TOKEN)
            for text in done:
                if take(text):
                    return stories, dropped
        take(carry)
    return stories, dropped


def split_stories(
    stories: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Seeded shuffle; the last `val_fraction` of stories (at least one) become validation."""
    if len(stories) < 2:
        raise ConfigError(f"need at least 2 stories to make a train/val split, got {len(stories)}")
    order = list(range(len(stories)))
    random.Random(seed).shuffle(order)
    n_val = min(max(1, round(len(stories) * val_fraction)), len(stories) - 1)
    val = [stories[i] for i in order[-n_val:]]
    train = [stories[i] for i in order[:-n_val]]
    return train, val


def stories_path(cfg: RunConfig) -> Path:
    """Where the raw story text lives (download writes it here, everything else reads it)."""
    return Path(cfg.data.data_dir) / "stories.txt"
