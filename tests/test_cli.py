from grim_reaper.cli import main


def test_bpe_demo_prints_merges(capsys):
    assert main(["bpe-demo"]) == 0
    out = capsys.readouterr().out
    assert "merge 1" in out
    assert "round trip: ok" in out


def test_bpe_demo_small_vocab_is_a_clear_error(capsys):
    assert main(["bpe-demo", "--vocab-size", "100"]) == 2
    assert "at least 257" in capsys.readouterr().err
