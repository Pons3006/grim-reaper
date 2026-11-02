"""Fetch the TinyStories text from the Hugging Face hub.

The file is pinned to a dataset revision so the same config always fetches the same text.
It is saved as `<data_dir>/stories.txt`, whatever it is called on the hub.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..config import RunConfig
from .stories import stories_path

DATASET_FILE = "TinyStories-train.txt"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


class DownloadError(RuntimeError):
    """Raised when the dataset cannot be fetched."""


def download_dataset(
    cfg: RunConfig, fetch: Callable[..., str] = hf_hub_download, force: bool = False
) -> Path:
    """Download the story text into cfg.data.data_dir and return its path.

    Skips the download if the file is already there, unless `force` is set. Safe to re-run.
    """
    target = stories_path(cfg)
    if target.is_file() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fetched = fetch(
            repo_id=cfg.data.dataset,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=str(target.parent),
        )
    except Exception as e:
        raise DownloadError(
            f"could not download {DATASET_FILE} from {cfg.data.dataset}: {e}"
        ) from e
    os.replace(fetched, target)
    return target
