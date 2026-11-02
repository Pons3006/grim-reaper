# Tokenizer and Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn TinyStories into `train.bin` / `val.bin` token files, with a from-scratch BPE tokenizer for learning and a Hugging Face `tokenizers` BPE for the real run.

**Architecture:** `tokenizer/bpe.py` is a pure-Python byte-level BPE. `tokenizer/train.py` trains the library BPE. A new `data/` package downloads text, splits stories deterministically, and tokenizes in chunks into flat `uint16` files plus `meta.json`. The CLI gains `bpe-demo`, `tokenizer train`, `data download`, `data prepare`.

**Tech Stack:** Python 3.10+, `tokenizers`, `huggingface_hub` (new), `numpy`, `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-11-02-tokenizer-and-data-pipeline-design.md`

## Global Constraints

- Python `>=3.10`; ruff line length 100, rules `E,F,I,UP,B`; `from __future__ import annotations` in every module, matching `config.py`.
- Vocab comes from `cfg.model.vocab_size` (4096 in both configs); special token is exactly `<|endoftext|>`.
- Token files are `uint16`; `prepare` asserts `vocab_size <= 65535`.
- Validation text never influences the tokenizer: it trains on the train split only.
- The split is a seeded shuffle of story indices (`cfg.train.seed`), last `val_fraction` become validation.
- No `datasets` library. Tests never touch the network.
- Config failures raise `ConfigError`; the CLI prints `error: ...` to stderr and returns 2, as `cli.py` does.
- One new dependency only: `huggingface_hub` in `pyproject.toml`.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Text containing emoji, accents, or invalid-looking bytes: encode then decode must return the original exactly.
2. Empty string: `encode("")` returns `[]`, `decode([])` returns `""`, no crash.
3. A story that contains the literal text `<|endoftext|>`: must not be silently split into extra stories by `prepare`, and the scratch BPE must treat it as the single special token.
4. A tiny `max_stories` with `val_fraction` that rounds to zero validation stories: must still yield at least one validation story, or fail with a clear message.
5. Re-running `download` and `prepare`: must skip finished work, and `--force` must redo it.

## File Structure

- Create `src/grim_reaper/tokenizer/bpe.py`: scratch `BPETokenizer`.
- Create `src/grim_reaper/tokenizer/train.py`: `train_tokenizer`, `load_tokenizer`.
- Create `src/grim_reaper/data/__init__.py`, `stories.py` (read + split), `download.py`, `prepare.py`.
- Modify `src/grim_reaper/cli.py`, `pyproject.toml`, `README.md`.
- Create tests: `tests/test_bpe.py`, `tests/test_tokenizer_train.py`, `tests/test_stories.py`, `tests/test_download.py`, `tests/test_prepare.py`.
- Create `docs/bpe-and-data.md`: teaching note, written task by task.

(`stories.py` is not in the spec's table. It holds the shared split so `tokenizer train` and `prepare` cannot disagree.)

---

### Task 1: Scratch byte-level BPE

**Files:**
- Create: `src/grim_reaper/tokenizer/bpe.py`
- Test: `tests/test_bpe.py`

**Interfaces:**
- Produces: `SPECIAL_TOKEN = "<|endoftext|>"`; `class BPETokenizer` with
  - `BPETokenizer.train(texts: Iterable[str], vocab_size: int, on_merge: Callable[[int, tuple[int, int], int, int], None] | None = None) -> BPETokenizer` (callback args: step, pair, new_id, count)
  - `encode(self, text: str) -> list[int]`, `decode(self, ids: list[int]) -> str`
  - `save(self, path: str | Path) -> None`, `@classmethod load(cls, path: str | Path) -> BPETokenizer`
  - attributes `merges: list[tuple[int, int]]`, `vocab_size: int`, `special_id: int` (the id of `SPECIAL_TOKEN`, equal to `vocab_size - 1`)

- [ ] **Step 1: Write failing tests in `tests/test_bpe.py`**
  - `test_base_ids_are_bytes`: trained with `vocab_size=257 + 1` on `["ab"]` gives zero merges; `encode("a") == [97]`.
  - `test_first_merge_is_most_frequent_pair`: corpus `["aaab aaab"]`, one merge allowed: `merges == [(97, 97)]`.
  - `test_training_is_deterministic`: two trainings on the same corpus give equal `merges`.
  - `test_round_trip_unicode`: texts `"héllo 🌍"`, `"naïve café"`, `"日本語"` all satisfy `decode(encode(t)) == t` (Review Focus 1).
  - `test_empty_string`: `encode("") == []`, `decode([]) == ""` (Review Focus 2).
  - `test_special_token_is_single_id`: `encode("a<|endoftext|>b")` contains `special_id` exactly once, and decodes back to the same string (Review Focus 3).
  - `test_merges_reduce_length`: after training on repeated text, `len(encode(text)) < len(text.encode())`.
  - `test_save_load_round_trip`: loaded tokenizer encodes identically.
  - `test_on_merge_called_per_merge`: callback count equals `len(merges)`.

- [ ] **Step 2: Run `pytest tests/test_bpe.py -v`; expect FAIL (import error).**

- [ ] **Step 3: Implement `BPETokenizer` in `bpe.py`.**
  - Pre-split with `re.compile(r" ?\w+| ?[^\w\s]+|\s+")`; split on `SPECIAL_TOKEN` first.
  - Ids `0-255` are bytes, merges get `256, 257, ...`, `special_id = vocab_size - 1`.
  - Training: count pieces, then repeat {count adjacent pairs weighted by piece frequency, pick max count with ties to the smaller pair, merge, record} until `len(merges) == vocab_size - 257` or no pair has count >= 2.
  - Encode applies merges in learned order per piece. Decode joins bytes and decodes with `errors="replace"`.
  - `save`/`load` use JSON (`{"vocab_size": ..., "merges": [[a, b], ...]}`).
  - Docstrings explain each step in teaching language; this file is read by the user.

- [ ] **Step 4: Run `pytest tests/test_bpe.py -v`; expect all PASS.**

- [ ] **Step 5: Commit** (`feat: add from-scratch byte-level BPE tokenizer`).

---

### Task 2: `bpe-demo` command and teaching note

**Files:**
- Modify: `src/grim_reaper/cli.py`
- Create: `docs/bpe-and-data.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `BPETokenizer.train(..., on_merge=...)`, `encode`, `decode`.
- Produces: subcommand `grim-reaper bpe-demo [--vocab-size N] [--text TEXT]`, exit code 0; default vocab size 270, default corpus a few built-in sentences.

- [ ] **Step 1: Write failing test `test_bpe_demo_prints_merges` in `tests/test_cli.py`:** `main(["bpe-demo"])` returns 0 and captured stdout contains `"merge 1"` and `"round trip: ok"`.
- [ ] **Step 2: Run it; expect FAIL (unknown command).**
- [ ] **Step 3: Add `_cmd_bpe_demo` and its parser in `cli.py`;** print each merge as readable text (the two pieces and the count), then the encoding of `--text` and the round-trip check. Start `docs/bpe-and-data.md` with a "What BPE does" section using the demo's output.
- [ ] **Step 4: Run `pytest tests/test_cli.py -v`; expect PASS.** Run `grim-reaper bpe-demo` by hand and confirm the output is readable.
- [ ] **Step 5: Commit** (`feat: add bpe-demo command and BPE teaching note`).

---

### Task 3: Story reading and deterministic split

**Files:**
- Create: `src/grim_reaper/data/__init__.py`, `src/grim_reaper/data/stories.py`
- Test: `tests/test_stories.py`

**Interfaces:**
- Produces:
  - `read_stories(path: str | Path, max_stories: int | None = None) -> tuple[list[str], int]`: returns kept stories and the count of dropped empty ones. Stories are separated by `<|endoftext|>`, stripped, empties dropped, reading stops once `max_stories` kept.
  - `split_stories(stories: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]`: returns `(train, val)`.

- [ ] **Step 1: Write failing tests:**
  - `test_read_stories_splits_and_drops_empty`: text `"a<|endoftext|>\n<|endoftext|>b"` gives `(["a", "b"], 1)`.
  - `test_read_stories_respects_max`: `max_stories=1` returns one story.
  - `test_split_is_deterministic`: same inputs and seed give equal results; a different seed gives a different order.
  - `test_split_no_overlap_and_complete`: `sorted(train + val) == sorted(stories)`.
  - `test_split_always_has_validation`: 3 stories with `val_fraction=0.01` still returns exactly 1 validation story (Review Focus 4); with 1 story, raises `ConfigError` saying at least 2 stories are needed.
- [ ] **Step 2: Run `pytest tests/test_stories.py -v`; expect FAIL.**
- [ ] **Step 3: Implement.** Use `random.Random(seed).shuffle` on an index list; validation size is `max(1, round(len * val_fraction))`. Read the file in streaming chunks so the whole dataset is never one string.
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** (`feat: add story reader and deterministic train/val split`).

---

### Task 4: Real tokenizer training

**Files:**
- Create: `src/grim_reaper/tokenizer/train.py`
- Modify: `src/grim_reaper/cli.py`
- Test: `tests/test_tokenizer_train.py`

**Interfaces:**
- Consumes: `read_stories`, `split_stories`, `RunConfig`.
- Produces:
  - `TOKENIZER_TRAIN_STORIES = 50_000`
  - `train_tokenizer(cfg: RunConfig, stories_path: str | Path) -> Path`: trains on the train split (capped at `TOKENIZER_TRAIN_STORIES`), writes to `cfg.data.tokenizer_path`, returns that path.
  - `load_tokenizer(path: str | Path) -> tokenizers.Tokenizer`; raises `ConfigError("tokenizer not found at ...; run `grim-reaper tokenizer train`")` if missing.
  - CLI: `grim-reaper tokenizer train CONFIG`.

- [ ] **Step 1: Write failing tests** (tiny inline text file, small `vocab_size`):
  - `test_train_writes_tokenizer_with_vocab_and_special`: `vocab_size` matches config and `<|endoftext|>` encodes to one id.
  - `test_trained_tokenizer_round_trips`: Review Focus 1 texts round-trip.
  - `test_validation_text_not_in_training`: a unique marker word placed only in a validation story does not appear in the vocabulary.
  - `test_load_missing_raises_config_error`.
  - `test_bridge_scratch_vs_library`: train both on the same sample; both round-trip exactly; assert both tokenize the sample to fewer tokens than raw bytes.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.** `tokenizers.Tokenizer(BPE())` with `ByteLevel` pre-tokenizer and decoder, `BpeTrainer(vocab_size=..., special_tokens=[SPECIAL_TOKEN], initial_alphabet=ByteLevel.alphabet())`. Create the parent directory. Add the `tokenizer train` subparser.
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Add the "Library BPE and the bridge" section to `docs/bpe-and-data.md`; commit** (`feat: train library byte-level BPE tokenizer`).

---

### Task 5: Download

**Files:**
- Create: `src/grim_reaper/data/download.py`
- Modify: `pyproject.toml` (add `huggingface_hub>=0.23`), `src/grim_reaper/cli.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Produces: `download_dataset(cfg: RunConfig, fetch: Callable[..., str] = hf_hub_download, force: bool = False) -> Path`: returns the path of the story text file inside `cfg.data.data_dir`. CLI: `grim-reaper data download CONFIG [--force]`.

- [ ] **Step 1: Write failing tests with a fake `fetch`:**
  - `test_download_copies_into_data_dir`.
  - `test_download_skips_existing`: second call does not call `fetch` (Review Focus 5).
  - `test_force_redownloads`.
  - `test_fetch_failure_raises_clear_error`: message names the file and repo.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.** Before coding, check the real file names on the hub for `roneneldan/TinyStories` (read-only lookup via `huggingface_hub.list_repo_files`) and record the chosen file and the dataset revision it resolved to in a module constant. Wrap hub exceptions in `DownloadError(RuntimeError)`; the CLI prints `error: ...`, returns 2.
- [ ] **Step 4: Run tests; expect PASS.** Then run `grim-reaper data download configs/debug.yaml` once for real and confirm the file exists.
- [ ] **Step 5: Commit** (`feat: add dataset download`).

---

### Task 6: Prepare token files

**Files:**
- Create: `src/grim_reaper/data/prepare.py`
- Modify: `src/grim_reaper/cli.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `read_stories`, `split_stories`, `load_tokenizer`, `SPECIAL_TOKEN`.
- Produces: `prepare_dataset(cfg: RunConfig, stories_path: str | Path, force: bool = False) -> dict`: writes `train.bin`, `val.bin`, `meta.json` into `cfg.data.data_dir` and returns the meta dict. CLI: `grim-reaper data prepare CONFIG [--force]`.

`meta.json` keys: `dataset`, `dataset_revision`, `source_sha256`, `tokenizer_sha256`, `vocab_size`, `seed`, `max_stories`, `val_fraction`, `stories` (`train`, `val`, `dropped_empty`), `tokens` (`train`, `val`), `versions` (`python`, `tokenizers`, `numpy`).

- [ ] **Step 1: Write failing tests** (tiny corpus, tokenizer from Task 4's trainer):
  - `test_bin_contents_match_tokenizer`: `train.bin` read as `uint16` equals the concatenation of each train story's ids followed by the end-of-text id.
  - `test_meta_fields_and_counts`: all keys present, token counts equal file lengths.
  - `test_vocab_mismatch_refuses`: tokenizer vocab differs from `model.vocab_size` raises `ConfigError`.
  - `test_missing_tokenizer_message`: says to run `tokenizer train`.
  - `test_story_with_literal_special_token_not_split_further`: Review Focus 3, story counts unchanged by an inner marker (it is treated as a single token inside one story only if the reader did not split on it; assert the documented behaviour).
  - `test_skip_when_unchanged_and_force_reruns`: Review Focus 5.
  - `test_vocab_above_uint16_refuses`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement.** Tokenize in batches with `encode_batch` and write with `array.tofile` appended per batch so memory stays flat. Skip when `meta.json` matches the current config, source hash, and tokenizer hash. Measure peak memory of the real full-dataset run in Task 7 and set the batch size from that measurement.
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** (`feat: add token file preparation with provenance metadata`).

---

### Task 7: End-to-end run and docs

**Files:**
- Modify: `README.md`, `docs/bpe-and-data.md`

- [ ] **Step 1: Run the full chain on `configs/debug.yaml`:** `grim-reaper data download configs/debug.yaml`, `grim-reaper tokenizer train configs/debug.yaml`, `grim-reaper data prepare configs/debug.yaml`. Expect three successes and `train.bin`, `val.bin`, `meta.json` in `data/tinystories/`.
- [ ] **Step 2: Inspect results:** print `meta.json`; decode the first 200 tokens of `train.bin` with the tokenizer and confirm it reads as English stories. Record the measured token counts and peak memory in `docs/bpe-and-data.md` (measured values only).
- [ ] **Step 3: Run `make test` and `make lint`; expect both clean.**
- [ ] **Step 4: Update the README** (Getting started commands, tick "dataset and tokenizer pipeline" in the Stage 1 roadmap line, repository layout) and finish the teaching note.
- [ ] **Step 5: Commit** (`docs: document tokenizer and data pipeline; mark pipeline step done`).
