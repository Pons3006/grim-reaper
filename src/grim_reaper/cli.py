"""Command-line entry point: `grim-reaper <command>`.

Commands are added stage by stage. Right now:
  config   validate a run config and print the resolved settings
"""

from __future__ import annotations

import argparse
import sys

import yaml

from . import __version__
from .config import ConfigError, load_config


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.path)
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False).rstrip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grim-reaper", description="Grim Reaper: a tiny LLM research studio."
    )
    parser.add_argument("--version", action="version", version=f"grim-reaper {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cfg = sub.add_parser("config", help="validate a run config and print it")
    p_cfg.add_argument("path", help="path to a YAML config, e.g. configs/debug.yaml")
    p_cfg.set_defaults(func=_cmd_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
