"""Command-line entry point: `grim-reaper <command>`.

Commands are added stage by stage. Right now:
  config    validate a run config and print the resolved settings
  bpe-demo  train the from-scratch BPE on a tiny corpus and show every merge
  data download     fetch the dataset text for a run config
  data prepare      tokenize the text into train.bin / val.bin / meta.json
  tokenizer train   train the real (Hugging Face) BPE tokenizer for a run config
"""

from __future__ import annotations

import argparse
import sys

import yaml

from . import __version__
from .config import ConfigError, load_config
from .data.download import DownloadError, download_dataset
from .data.prepare import prepare_dataset
from .data.stories import stories_path
from .tokenizer.bpe import BPETokenizer
from .tokenizer.train import train_tokenizer


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.path)
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False).rstrip())
    return 0


DEMO_CORPUS = [
    "once upon a time there was a little cat. the cat liked to play with the ball.",
    "the little dog saw the cat and the cat saw the dog. they liked to play.",
    "one day the ball rolled down the hill and the cat and the dog ran after it.",
]


def _show(piece: bytes) -> str:
    return repr(piece.decode("utf-8", errors="replace"))


def _cmd_bpe_demo(args: argparse.Namespace) -> int:
    texts = [args.text] if args.text else DEMO_CORPUS
    token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    print(f"training on {len(texts)} text(s), vocab size {args.vocab_size}\n")

    def on_merge(step: int, pair: tuple[int, int], new_id: int, count: int) -> None:
        token_bytes[new_id] = token_bytes[pair[0]] + token_bytes[pair[1]]
        print(
            f"merge {step}: {_show(token_bytes[pair[0]])} + {_show(token_bytes[pair[1]])}"
            f" -> id {new_id} {_show(token_bytes[new_id])}  (seen {count}x)"
        )

    tok = BPETokenizer.train(texts, args.vocab_size, on_merge=on_merge)
    sample = texts[0]
    ids = tok.encode(sample)
    print(f"\nencoding: {sample!r}")
    print(f"  {len(sample.encode())} bytes -> {len(ids)} tokens: {ids}")
    ok = tok.decode(ids) == sample
    print(f"round trip: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


def _cmd_tokenizer_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.path)
    src = stories_path(cfg)
    if not src.is_file():
        raise ConfigError(f"no story text at {src}; run `grim-reaper data download {args.path}`")
    out = train_tokenizer(cfg, src)
    print(f"wrote {out} (vocab size {cfg.model.vocab_size})")
    return 0


def _cmd_data_download(args: argparse.Namespace) -> int:
    cfg = load_config(args.path)
    path = download_dataset(cfg, force=args.force)
    print(f"story text ready at {path}")
    return 0


def _cmd_data_prepare(args: argparse.Namespace) -> int:
    cfg = load_config(args.path)
    src = stories_path(cfg)
    if not src.is_file():
        raise ConfigError(f"no story text at {src}; run `grim-reaper data download {args.path}`")
    meta = prepare_dataset(cfg, src, force=args.force)
    s, t = meta["stories"], meta["tokens"]
    print(f"stories: {s['train']} train / {s['val']} val ({s['dropped_empty']} empty dropped)")
    print(f"tokens:  {t['train']} train / {t['val']} val  -> {cfg.data.data_dir}")
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

    p_demo = sub.add_parser("bpe-demo", help="watch the from-scratch BPE learn merges")
    p_demo.add_argument("--vocab-size", type=int, default=270, help="default: 270")
    p_demo.add_argument("--text", help="train on this text instead of the built-in corpus")
    p_demo.set_defaults(func=_cmd_bpe_demo)

    p_tok = sub.add_parser("tokenizer", help="tokenizer commands")
    tok_sub = p_tok.add_subparsers(dest="tokenizer_command", required=True)
    p_train = tok_sub.add_parser("train", help="train the real BPE tokenizer for a run config")
    p_train.add_argument("path", help="path to a YAML config, e.g. configs/debug.yaml")
    p_train.set_defaults(func=_cmd_tokenizer_train)

    p_data = sub.add_parser("data", help="dataset commands")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p_dl = data_sub.add_parser("download", help="download the dataset text")
    p_dl.add_argument("path", help="path to a YAML config, e.g. configs/debug.yaml")
    p_dl.add_argument("--force", action="store_true", help="download again even if present")
    p_dl.set_defaults(func=_cmd_data_download)
    p_prep = data_sub.add_parser("prepare", help="tokenize the text into train.bin / val.bin")
    p_prep.add_argument("path", help="path to a YAML config, e.g. configs/debug.yaml")
    p_prep.add_argument("--force", action="store_true", help="redo even if up to date")
    p_prep.set_defaults(func=_cmd_data_prepare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
