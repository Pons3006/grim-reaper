"""Turn story text into flat token files the training loop can read.

Output, in `cfg.data.data_dir`:
  train.bin / val.bin   uint16 token ids; each story is followed by the end-of-text id
  meta.json             provenance: what text, which tokenizer, which split, how many tokens

`uint16` works because the vocab is far below 65535, and it keeps the files small. The training
loop can later `np.memmap` them and slice random windows without loading everything.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import tokenizers

from ..config import ConfigError, RunConfig
from ..tokenizer.bpe import SPECIAL_TOKEN
from ..tokenizer.train import load_tokenizer
from .download import DATASET_REVISION
from .stories import read_stories, split_stories

MAX_UINT16 = 65535
BATCH_STORIES = 2000  # stories encoded per call; keeps memory flat on large datasets


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _write_split(path: Path, stories: list[str], tok: tokenizers.Tokenizer, eot: int) -> int:
    """Tokenize `stories` batch by batch and append to `path`. Returns the token count."""
    total = 0
    with open(path, "wb") as f:
        for start in range(0, len(stories), BATCH_STORIES):
            batch = tok.encode_batch(stories[start : start + BATCH_STORIES])
            ids = [i for enc in batch for i in (*enc.ids, eot)]
            np.asarray(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
    return total


def prepare_dataset(
    cfg: RunConfig, stories_path: str | Path, force: bool = False
) -> dict[str, Any]:
    """Write train.bin, val.bin and meta.json; return the metadata.

    Does nothing (and returns the existing metadata) if the files already match this config,
    source text and tokenizer, unless `force` is set.
    """
    if cfg.model.vocab_size > MAX_UINT16:
        raise ConfigError(
            f"model.vocab_size {cfg.model.vocab_size} does not fit uint16 token files "
            f"(max {MAX_UINT16})"
        )
    tok = load_tokenizer(cfg.data.tokenizer_path)
    if tok.get_vocab_size() != cfg.model.vocab_size:
        raise ConfigError(
            f"tokenizer vocab size ({tok.get_vocab_size()}) does not match "
            f"model.vocab_size ({cfg.model.vocab_size}); retrain with `grim-reaper tokenizer train`"
        )

    out = Path(cfg.data.data_dir)
    src = Path(stories_path)
    identity = {
        "dataset": cfg.data.dataset,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": _sha256(src),
        "tokenizer_sha256": _sha256(Path(cfg.data.tokenizer_path)),
        "vocab_size": cfg.model.vocab_size,
        "seed": cfg.train.seed,
        "max_stories": cfg.data.max_stories,
        "val_fraction": cfg.data.val_fraction,
    }
    meta_path = out / "meta.json"
    if (
        not force
        and meta_path.is_file()
        and (out / "train.bin").is_file()
        and (out / "val.bin").is_file()
    ):
        existing = json.loads(meta_path.read_text())
        if all(existing.get(k) == v for k, v in identity.items()):
            return existing

    stories, dropped = read_stories(src, cfg.data.max_stories)
    train, val = split_stories(stories, cfg.data.val_fraction, cfg.train.seed)
    eot = tok.token_to_id(SPECIAL_TOKEN)

    out.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)  # a half-written run must not look finished
    meta = {
        **identity,
        "stories": {"train": len(train), "val": len(val), "dropped_empty": dropped},
        "tokens": {
            "train": _write_split(out / "train.bin", train, tok, eot),
            "val": _write_split(out / "val.bin", val, tok, eot),
        },
        "versions": {
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
