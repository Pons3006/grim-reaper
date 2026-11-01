"""Run configuration: typed, validated, and loadable from YAML.

One YAML file describes one run. The resolved config is saved next to the run's
outputs so any result can be traced back to the exact settings that produced it.
Unknown keys are rejected, so a typo in a config fails loudly instead of being ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

DEVICES = ("auto", "cuda", "mps", "cpu")
DTYPES = ("auto", "float32", "bfloat16", "float16")

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a config file is malformed or has invalid values."""


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 256  # context length
    n_layer: int = 5
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False

    def validate(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"model.{name} must be > 0, got {getattr(self, name)}")
        if self.n_embd % self.n_head != 0:
            raise ConfigError(
                f"model.n_embd ({self.n_embd}) must be divisible by model.n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"model.dropout must be in [0, 1), got {self.dropout}")


@dataclass
class DataConfig:
    dataset: str = "roneneldan/TinyStories"
    data_dir: str = "data/tinystories"
    tokenizer_path: str = "data/tinystories/tokenizer.json"
    val_fraction: float = 0.01
    max_stories: int | None = None  # None = use everything; set small for quick tests

    def validate(self) -> None:
        if not 0.0 < self.val_fraction < 0.5:
            raise ConfigError(f"data.val_fraction must be in (0, 0.5), got {self.val_fraction}")
        if self.max_stories is not None and self.max_stories <= 0:
            raise ConfigError(f"data.max_stories must be > 0 or null, got {self.max_stories}")


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 5000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    sample_interval: int = 500
    checkpoint_interval: int = 500
    device: str = "auto"
    dtype: str = "auto"
    out_dir: str = "experiments/run"

    def validate(self) -> None:
        for name in (
            "batch_size",
            "grad_accum_steps",
            "max_steps",
            "eval_interval",
            "eval_iters",
            "log_interval",
            "sample_interval",
            "checkpoint_interval",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"train.{name} must be > 0, got {getattr(self, name)}")
        if self.lr <= 0 or self.min_lr < 0 or self.min_lr > self.lr:
            raise ConfigError(
                f"need 0 <= train.min_lr <= train.lr, got min_lr={self.min_lr}, lr={self.lr}"
            )
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ConfigError(
                f"train.warmup_steps must be in [0, max_steps), got {self.warmup_steps}"
            )
        if self.device not in DEVICES:
            raise ConfigError(f"train.device must be one of {DEVICES}, got {self.device!r}")
        if self.dtype not in DTYPES:
            raise ConfigError(f"train.dtype must be one of {DTYPES}, got {self.dtype!r}")


@dataclass
class RunConfig:
    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ConfigError("name must not be empty")
        self.model.validate()
        self.data.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type[T], raw: Any, path: str) -> T:
    """Build dataclass `cls` from a mapping, rejecting unknown keys and wrong types."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"'{path}' must be a mapping, got {type(raw).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        where = path or "top level"
        raise ConfigError(f"unknown key(s) at {where}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        f = fields[name]
        key = f"{path}.{name}" if path else name
        if dataclasses.is_dataclass(_resolve(f)):
            kwargs[name] = _build(_resolve(f), value, key)
        else:
            kwargs[name] = _coerce(f, value, key)
    return cls(**kwargs)


def _resolve(f: dataclasses.Field) -> Any:
    """Return the dataclass type of a nested config field, else the raw annotation."""
    nested = {"ModelConfig": ModelConfig, "DataConfig": DataConfig, "TrainConfig": TrainConfig}
    t = f.type
    return nested.get(t, t) if isinstance(t, str) else t


def _coerce(f: dataclasses.Field, value: Any, key: str) -> Any:
    ann = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    nullable = "None" in ann
    if value is None:
        if nullable:
            return None
        raise ConfigError(f"'{key}' must not be null")
    base = ann.replace(" | None", "").strip()
    if base == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{key}' must be an integer, got {value!r}")
        return value
    if base == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{key}' must be a number, got {value!r}")
        return float(value)
    if base == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"'{key}' must be true or false, got {value!r}")
        return value
    if base == "str":
        if not isinstance(value, str):
            raise ConfigError(f"'{key}' must be a string, got {value!r}")
        return value
    return value


def config_from_dict(raw: dict[str, Any]) -> RunConfig:
    cfg = _build(RunConfig, raw, "")
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run config from a YAML file."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"could not parse {p}: {e}") from e
    if raw is None:
        raw = {}
    return config_from_dict(raw)


def save_config(cfg: RunConfig, path: str | Path) -> None:
    """Write the resolved config (all defaults filled in) so a run is reproducible."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
