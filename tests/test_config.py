from pathlib import Path

import pytest
import yaml

from grim_reaper.cli import main
from grim_reaper.config import (
    ConfigError,
    RunConfig,
    config_from_dict,
    load_config,
    save_config,
)

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


def test_shipped_configs_exist():
    assert {p.stem for p in CONFIGS} >= {"debug", "small"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.model.n_embd % cfg.model.n_head == 0


def test_defaults_are_valid():
    RunConfig().validate()


def test_partial_config_fills_defaults():
    cfg = config_from_dict({"name": "x", "model": {"n_layer": 2}})
    assert cfg.model.n_layer == 2
    assert cfg.model.n_head == RunConfig().model.n_head
    assert cfg.train.seed == 1337


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown key.*n_layers"):
        config_from_dict({"model": {"n_layers": 4}})


def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError, match="unknown key.*modle"):
        config_from_dict({"modle": {}})


def test_wrong_type_rejected():
    with pytest.raises(ConfigError, match="model.n_layer.*integer"):
        config_from_dict({"model": {"n_layer": "four"}})
    with pytest.raises(ConfigError, match="bias.*true or false"):
        config_from_dict({"model": {"bias": "yes"}})


def test_bool_not_accepted_as_int():
    with pytest.raises(ConfigError, match="integer"):
        config_from_dict({"model": {"n_layer": True}})


def test_int_accepted_for_float():
    cfg = config_from_dict({"train": {"lr": 1, "min_lr": 0}})
    assert cfg.train.lr == 1.0 and isinstance(cfg.train.lr, float)


def test_null_only_where_allowed():
    assert config_from_dict({"data": {"max_stories": None}}).data.max_stories is None
    with pytest.raises(ConfigError, match="must not be null"):
        config_from_dict({"model": {"n_layer": None}})


def test_heads_must_divide_embedding():
    with pytest.raises(ConfigError, match="divisible"):
        config_from_dict({"model": {"n_embd": 100, "n_head": 6}})


def test_lr_ordering_and_warmup():
    with pytest.raises(ConfigError, match="min_lr"):
        config_from_dict({"train": {"lr": 1e-4, "min_lr": 1e-3}})
    with pytest.raises(ConfigError, match="warmup_steps"):
        config_from_dict({"train": {"max_steps": 100, "warmup_steps": 100}})


def test_bad_device_and_dtype():
    with pytest.raises(ConfigError, match="device"):
        config_from_dict({"train": {"device": "tpu"}})
    with pytest.raises(ConfigError, match="dtype"):
        config_from_dict({"train": {"dtype": "int4"}})


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model: [unclosed")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(p)


def test_empty_file_gives_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_config(p) == RunConfig()


def test_save_load_roundtrip(tmp_path):
    cfg = load_config(CONFIGS[0])
    out = tmp_path / "nested" / "resolved.yaml"
    save_config(cfg, out)
    assert load_config(out) == cfg


def test_cli_config_prints_resolved(capsys):
    assert main(["config", str(CONFIGS[0])]) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed == load_config(CONFIGS[0]).to_dict()


def test_cli_config_error_exit_code(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  n_embd: 100\n  n_head: 6\n")
    assert main(["config", str(p)]) == 2
    assert "config error" in capsys.readouterr().err
