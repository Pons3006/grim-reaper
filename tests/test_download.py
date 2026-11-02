from pathlib import Path

import pytest

from grim_reaper.config import config_from_dict
from grim_reaper.data.download import DATASET_FILE, DownloadError, download_dataset
from grim_reaper.data.stories import stories_path


def _cfg(tmp_path):
    return config_from_dict({"data": {"data_dir": str(tmp_path / "data")}})


class FakeFetch:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, repo_id, filename, repo_type, revision, local_dir):
        self.calls += 1
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("story one<|endoftext|>story two")
        return str(target)


def test_download_puts_text_at_stories_path(tmp_path):
    cfg = _cfg(tmp_path)
    path = download_dataset(cfg, fetch=FakeFetch())
    assert path == stories_path(cfg)
    assert path.read_text().startswith("story one")
    assert not (path.parent / DATASET_FILE).exists()


def test_download_skips_existing(tmp_path):
    cfg = _cfg(tmp_path)
    fetch = FakeFetch()
    download_dataset(cfg, fetch=fetch)
    download_dataset(cfg, fetch=fetch)
    assert fetch.calls == 1


def test_force_redownloads(tmp_path):
    cfg = _cfg(tmp_path)
    fetch = FakeFetch()
    download_dataset(cfg, fetch=fetch)
    download_dataset(cfg, fetch=fetch, force=True)
    assert fetch.calls == 2


def test_fetch_failure_raises_clear_error(tmp_path):
    def broken(**kwargs):
        raise OSError("network down")

    with pytest.raises(DownloadError, match=rf"{DATASET_FILE}.*roneneldan/TinyStories"):
        download_dataset(_cfg(tmp_path), fetch=broken)
