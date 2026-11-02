import pytest

from grim_reaper.config import ConfigError
from grim_reaper.data.stories import read_stories, split_stories


def test_read_stories_splits_and_drops_empty(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("a<|endoftext|>\n<|endoftext|>b")
    assert read_stories(path) == (["a", "b"], 1)


def test_read_stories_respects_max(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("a<|endoftext|>b<|endoftext|>c")
    stories, _ = read_stories(path, max_stories=1)
    assert stories == ["a"]


def test_read_stories_handles_separator_across_chunk_boundary(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("one<|endoftext|>two<|endoftext|>three")
    stories, _ = read_stories(path, chunk_chars=5)
    assert stories == ["one", "two", "three"]


def test_split_is_deterministic():
    stories = [str(i) for i in range(100)]
    assert split_stories(stories, 0.1, seed=1) == split_stories(stories, 0.1, seed=1)
    assert split_stories(stories, 0.1, seed=1) != split_stories(stories, 0.1, seed=2)


def test_split_no_overlap_and_complete():
    stories = [str(i) for i in range(100)]
    train, val = split_stories(stories, 0.1, seed=1)
    assert len(val) == 10
    assert sorted(train + val) == sorted(stories)


def test_split_always_has_validation():
    train, val = split_stories(["a", "b", "c"], 0.01, seed=1)
    assert len(val) == 1 and len(train) == 2


def test_split_single_story_raises():
    with pytest.raises(ConfigError, match="at least 2 stories"):
        split_stories(["only"], 0.1, seed=1)
