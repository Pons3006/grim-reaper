import pytest

from grim_reaper.config import ConfigError, config_from_dict
from grim_reaper.data.stories import read_stories, split_stories
from grim_reaper.tokenizer.bpe import SPECIAL_TOKEN, BPETokenizer
from grim_reaper.tokenizer.train import load_tokenizer, train_tokenizer

WORDS = "the cat sat on a mat and the dog ran to the big red ball in the sun".split()


def _stories(n=60):
    # Numbered so every story is unique and the corpus has enough varied pairs to fill the vocab.
    return [
        " ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(12)) + f" story {i * 37}."
        for i in range(n)
    ]


def _write(tmp_path, stories):
    path = tmp_path / "stories.txt"
    path.write_text(SPECIAL_TOKEN.join(stories))
    return path


def _cfg(tmp_path, vocab=300):
    return config_from_dict(
        {
            "model": {"vocab_size": vocab},
            "data": {
                "data_dir": str(tmp_path),
                "tokenizer_path": str(tmp_path / "out" / "tokenizer.json"),
                "val_fraction": 0.1,
            },
        }
    )


def test_train_writes_tokenizer_with_vocab_and_special(tmp_path):
    cfg = _cfg(tmp_path)
    path = train_tokenizer(cfg, _write(tmp_path, _stories()))
    tok = load_tokenizer(path)
    assert tok.get_vocab_size() == 300
    assert len(tok.encode(SPECIAL_TOKEN).ids) == 1


def test_trained_tokenizer_round_trips(tmp_path):
    cfg = _cfg(tmp_path)
    tok = load_tokenizer(train_tokenizer(cfg, _write(tmp_path, _stories())))
    for text in ["héllo 🌍", "naïve café", "日本語", "the cat sat"]:
        assert tok.decode(tok.encode(text).ids) == text


def test_validation_text_not_in_training(tmp_path):
    cfg = _cfg(tmp_path)
    stories = _stories()
    _, val = split_stories(stories, cfg.data.val_fraction, cfg.train.seed)
    stories[stories.index(val[0])] = "zxqv " * 50
    tok = load_tokenizer(train_tokenizer(cfg, _write(tmp_path, stories)))
    assert not any("zx" in token for token in tok.get_vocab())


def test_load_missing_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="tokenizer train"):
        load_tokenizer(tmp_path / "nope.json")


def test_bridge_scratch_vs_library(tmp_path):
    cfg = _cfg(tmp_path)
    stories = _stories()
    path = _write(tmp_path, stories)
    library = load_tokenizer(train_tokenizer(cfg, path))
    train, _ = split_stories(read_stories(path)[0], cfg.data.val_fraction, cfg.train.seed)
    scratch = BPETokenizer.train(train, vocab_size=300)
    sample = stories[0]
    raw = len(sample.encode())
    for ids, text in [
        (library.encode(sample).ids, library.decode(library.encode(sample).ids)),
        (scratch.encode(sample), scratch.decode(scratch.encode(sample))),
    ]:
        assert text == sample
        assert len(ids) < raw


def test_train_writes_split_sidecar(tmp_path):
    import json

    cfg = _cfg(tmp_path)
    path = train_tokenizer(cfg, _write(tmp_path, _stories()))
    side = json.loads(path.with_suffix(".meta.json").read_text())
    assert side["max_stories"] is None and side["seed"] == cfg.train.seed
    assert side["val_fraction"] == 0.1 and side["vocab_size"] == 300 and "source_sha256" in side


def test_vocab_below_257_refuses(tmp_path):
    with pytest.raises(ConfigError, match="at least 257"):
        train_tokenizer(_cfg(tmp_path, vocab=100), _write(tmp_path, _stories()))


def test_corpus_too_small_for_vocab_refuses(tmp_path):
    with pytest.raises(ConfigError, match="corpus is too small"):
        train_tokenizer(_cfg(tmp_path, vocab=4096), _write(tmp_path, _stories()))


def test_load_corrupt_raises_config_error(tmp_path):
    bad = tmp_path / "tokenizer.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError, match="unreadable.*tokenizer train"):
        load_tokenizer(bad)
