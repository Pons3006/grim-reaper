# Tokenizer and data pipeline: design

Stage 1, step 2 of Grim Reaper. Turns the TinyStories text into token files the training loop can read.

## Goal

- Learn how byte-level BPE works by writing it from scratch (small data), then use Hugging Face `tokenizers` for the real run.
- Produce `train.bin` and `val.bin` (flat `uint16` token arrays) plus `meta.json` provenance.
- Out of scope: batch loader, model, training loop (next steps).

## Already fixed by the existing configs

Dataset `roneneldan/TinyStories`, `vocab_size: 4096`, `data_dir`, `tokenizer_path`, `val_fraction`, `max_stories`, `train.seed`. Hardware target: Apple M2, 8 GB RAM, so all processing is streamed or chunked.

## Components

| Unit | Purpose |
|---|---|
| `tokenizer/bpe.py` | Pure-Python byte-level BPE (`train`, `encode`, `decode`, `save`, `load`). Learning version, for a few thousand stories and a small vocab. |
| `tokenizer/train.py` | Trains a `tokenizers` byte-level BPE (vocab from config, special token `<\|endoftext\|>`), writes `tokenizer.json`. |
| `data/download.py` | Fetches TinyStories text into `data_dir` via `huggingface_hub`. Skips files already present. Honours `max_stories`. |
| `data/prepare.py` | Splits stories into train/val, tokenizes in chunks, appends the end-of-text id after every story, writes `train.bin`, `val.bin`, `meta.json`. |
| `cli.py` | `tokenizer train`, `data download`, `data prepare`, `bpe-demo`. |

## Scratch BPE (`bpe.py`)

1. Text is split into pieces by a simple regex (words with an optional leading space, punctuation runs, whitespace) so merges never cross word boundaries. The special token is split out first and never merged.
2. Each piece becomes its UTF-8 bytes, so ids 0-255 are the base vocab and any text is encodable.
3. Training repeats until the vocab is full: count adjacent id pairs across all pieces (weighted by piece frequency), merge the most frequent pair into a new id, record the merge in order. Ties break on the smaller pair, so training is deterministic.
4. Encoding applies the learned merges in the order they were learned. Decoding concatenates bytes and decodes UTF-8 with `errors="replace"`.
5. `bpe-demo` prints each merge as it is learned, on a tiny corpus.

## Real tokenizer (`train.py`)

`tokenizers` `ByteLevel` pre-tokenizer and decoder, `BpeTrainer(vocab_size=cfg.model.vocab_size, special_tokens=["<|endoftext|>"])`. Trained on the **train split only**, up to a capped number of stories, so validation text never influences the vocabulary. Saves to `data.tokenizer_path`.

## Data flow

```
download -> data_dir/*.txt -> split stories (seeded) -> train tokenizer (train stories)
         -> prepare: tokenize chunk by chunk -> train.bin / val.bin / meta.json
```

- Stories are separated by `<|endoftext|>` in the source text. Empty stories are dropped and counted.
- The split is a seeded shuffle (`train.seed`) of story indices; the last `val_fraction` become validation. Same config gives the same split.
- Tokens are written as `uint16` (valid because vocab is 4096; `prepare` asserts `vocab_size <= 65535`).
- `meta.json` records dataset name and revision, file hashes, tokenizer hash, vocab size, story counts (train, val, dropped), token counts, seed, `max_stories`, and library versions.

## Bridge test

On one small sample, train both tokenizers and check that each round-trips the text exactly, and report vocab size and tokens per story side by side. This is a teaching check, not a claim that the two produce identical merges.

## Errors

Every failure is a clear message, not a traceback, using the existing `ConfigError` style where config is at fault:

- Tokenizer vocab size differs from `model.vocab_size`: refuse to prepare.
- `tokenizer_path` missing when `prepare` runs: tell the user to run `tokenizer train`.
- Network or hub failure on download: report which file, keep partial progress, safe to re-run.
- Source file missing or empty after filtering: stop with the counts.
- Existing `train.bin` and `val.bin` with a matching `meta.json`: skip unless `--force`.

## Testing

Unit tests on tiny inline corpora, no network: BPE merge order and determinism, encode/decode round-trip including emoji and unseen bytes, special token never split, split determinism and no train/val overlap, `uint16` file contents match the tokenizer output, `meta.json` fields, vocab mismatch error. Download is tested against a local fake file, not the hub.

## Dependencies

Add `huggingface_hub` to `pyproject.toml`. No `datasets` library.

## Open items to verify during implementation

- Exact TinyStories file names on the hub (V1 vs V2 train/valid text files) and whether to use the provided validation file or our own split. The design above uses our own seeded split of the train file so `val_fraction` keeps meaning.
- Peak memory of `prepare` on the full dataset on 8 GB; chunk size to be set from a measurement, not a guess.
