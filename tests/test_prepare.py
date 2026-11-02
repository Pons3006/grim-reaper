import json

import numpy as np
import pytest

from grim_reaper.config import ConfigError, config_from_dict
from grim_reaper.data.prepare import prepare_dataset
from grim_reaper.data.stories import read_stories, split_stories
from grim_reaper.tokenizer.bpe import SPECIAL_TOKEN
from grim_reaper.tokenizer.train import load_tokenizer, train_tokenizer

WORDS = "the cat sat on a mat and the dog ran to the big red ball in the sun".split()


def _stories(n=60):
    return [
        " ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(12)) + f" story {i * 37}."
        for i in range(n)
    ]


def _setup(tmp_path, stories=None, vocab=300):
    cfg = config_from_dict(
        {
            "model": {"vocab_size": vocab},
            "data": {
                "data_dir": str(tmp_path),
                "tokenizer_path": str(tmp_path / "tokenizer.json"),
                "val_fraction": 0.1,
            },
        }
    )
    src = tmp_path / "stories.txt"
    src.write_text(SPECIAL_TOKEN.join(stories or _stories()))
    train_tokenizer(cfg, src)
    return cfg, src


def test_bin_contents_match_tokenizer(tmp_path):
    cfg, src = _setup(tmp_path)
    prepare_dataset(cfg, src)
    tok = load_tokenizer(cfg.data.tokenizer_path)
    eot = tok.token_to_id(SPECIAL_TOKEN)
    train, val = split_stories(read_stories(src)[0], cfg.data.val_fraction, cfg.train.seed)
    for name, group in [("train", train), ("val", val)]:
        expected = [i for s in group for i in [*tok.encode(s).ids, eot]]
        got = np.fromfile(tmp_path / f"{name}.bin", dtype=np.uint16)
        assert got.tolist() == expected


def test_meta_fields_and_counts(tmp_path):
    cfg, src = _setup(tmp_path)
    meta = prepare_dataset(cfg, src)
    assert json.loads((tmp_path / "meta.json").read_text()) == meta
    for key in [
        "dataset",
        "dataset_revision",
        "source_sha256",
        "tokenizer_sha256",
        "vocab_size",
        "seed",
        "max_stories",
        "val_fraction",
        "stories",
        "tokens",
        "versions",
    ]:
        assert key in meta
    assert meta["stories"]["train"] + meta["stories"]["val"] == 60
    assert meta["tokens"]["train"] == (tmp_path / "train.bin").stat().st_size // 2
    assert meta["tokens"]["val"] == (tmp_path / "val.bin").stat().st_size // 2


def test_vocab_mismatch_refuses(tmp_path):
    cfg, src = _setup(tmp_path)
    cfg.model.vocab_size = 400
    with pytest.raises(ConfigError, match="vocab"):
        prepare_dataset(cfg, src)


def test_missing_tokenizer_message(tmp_path):
    cfg, src = _setup(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(ConfigError, match="tokenizer train"):
        prepare_dataset(cfg, src)


def test_vocab_above_uint16_refuses(tmp_path):
    cfg, src = _setup(tmp_path)
    cfg.model.vocab_size = 70000
    with pytest.raises(ConfigError, match="65535"):
        prepare_dataset(cfg, src)


def test_special_token_in_text_is_a_story_boundary(tmp_path):
    stories = _stories()
    stories[0] = "left half" + SPECIAL_TOKEN + "right half"
    cfg, src = _setup(tmp_path, stories)
    meta = prepare_dataset(cfg, src)
    assert meta["stories"]["train"] + meta["stories"]["val"] == 61


def test_skip_when_unchanged_and_force_reruns(tmp_path):
    cfg, src = _setup(tmp_path)
    prepare_dataset(cfg, src)
    first = (tmp_path / "train.bin").stat().st_mtime_ns
    prepare_dataset(cfg, src)
    assert (tmp_path / "train.bin").stat().st_mtime_ns == first
    prepare_dataset(cfg, src, force=True)
    assert (tmp_path / "train.bin").stat().st_mtime_ns != first


def test_tokenizer_trained_for_different_split_refuses(tmp_path):
    cfg, src = _setup(tmp_path)
    cfg.data.max_stories = 30
    with pytest.raises(ConfigError, match="different split.*tokenizer train"):
        prepare_dataset(cfg, src)


def test_tokenizer_without_sidecar_refuses(tmp_path):
    cfg, src = _setup(tmp_path)
    (tmp_path / "tokenizer.meta.json").unlink()
    with pytest.raises(ConfigError, match="tokenizer train"):
        prepare_dataset(cfg, src)


def test_truncated_meta_is_treated_as_stale(tmp_path):
    cfg, src = _setup(tmp_path)
    prepare_dataset(cfg, src)
    (tmp_path / "meta.json").write_text('{"dataset": ')
    meta = prepare_dataset(cfg, src)
    assert meta["stories"]["train"] + meta["stories"]["val"] == 60
    assert json.loads((tmp_path / "meta.json").read_text()) == meta


@pytest.mark.parametrize(
    "change",
    [
        lambda cfg, src: setattr(cfg.data, "max_stories", 50),
        lambda cfg, src: setattr(cfg.train, "seed", 7),
        lambda cfg, src: setattr(cfg.data, "val_fraction", 0.2),
        lambda cfg, src: src.write_text(SPECIAL_TOKEN.join(_stories(80))),
    ],
    ids=["max_stories", "seed", "val_fraction", "source_text"],
)
def test_rerun_rewrites_when_input_changes(tmp_path, change):
    cfg, src = _setup(tmp_path)
    prepare_dataset(cfg, src)
    first = (tmp_path / "train.bin").stat().st_mtime_ns
    change(cfg, src)
    train_tokenizer(cfg, src)  # the tokenizer must be retrained for the new split
    prepare_dataset(cfg, src)
    assert (tmp_path / "train.bin").stat().st_mtime_ns != first


def test_rerun_rewrites_when_tokenizer_file_changes(tmp_path):
    cfg, src = _setup(tmp_path)
    prepare_dataset(cfg, src)
    first = (tmp_path / "train.bin").stat().st_mtime_ns
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=1))  # same tokenizer, new bytes
    prepare_dataset(cfg, src)
    assert (tmp_path / "train.bin").stat().st_mtime_ns != first
