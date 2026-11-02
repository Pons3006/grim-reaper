from grim_reaper.cli import main


def test_bpe_demo_prints_merges(capsys):
    assert main(["bpe-demo"]) == 0
    out = capsys.readouterr().out
    assert "merge 1" in out
    assert "round trip: ok" in out
