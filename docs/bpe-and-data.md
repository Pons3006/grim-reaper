# BPE and the data pipeline

A guided tour of how Grim Reaper turns text into the numbers a model trains on.

## What BPE does

A model cannot read letters; it reads integers. A **tokenizer** maps text to integer ids and back.
We want a vocabulary that is small (cheap to train) but still keeps common words short.

**Byte-pair encoding (BPE)** gets there by merging:

1. Start with raw UTF-8 bytes. Ids 0-255 are the byte values, so any text, including emoji and
   other languages, can always be encoded.
2. Count how often each adjacent pair of ids occurs.
3. Give the most frequent pair a new id and replace every occurrence.
4. Repeat until the vocabulary reaches the target size.

The ordered list of merges is the whole tokenizer. Encoding replays the merges in the order they
were learned.

Try it: `grim-reaper bpe-demo` trains our from-scratch tokenizer (`src/grim_reaper/tokenizer/bpe.py`)
on three tiny sentences and prints every merge:

```
merge 1: ' ' + 't' -> id 256 ' t'  (seen 14x)
merge 2: 'h' + 'e' -> id 257 'he'  (seen 12x)
merge 3: ' t' + 'he' -> id 258 ' the'  (seen 11x)
merge 8: ' c' + 'at' -> id 263 ' cat'  (seen 5x)
```

Notice that merges build on earlier merges: `' t'` and `'he'` combine into `' the'`, and
`' c'` and `'at'` into `' cat'`. Frequent words end up as a single token.

Details worth knowing:

- **Pre-splitting.** Text is first cut into words, punctuation and whitespace, so a merge never
  spans two words.
- **Determinism.** Ties between equally frequent pairs go to the smaller pair, so the same corpus
  always gives the same tokenizer.
- **Stopping early.** If no pair repeats, merging would only memorise the corpus, so training stops.
- **Special token.** `<|endoftext|>` is a single id (the last one) that is never merged or split.
  We use it to mark where one story ends and the next begins.
- **Vocab arithmetic.** vocab = 256 bytes + merges + 1 special token, so a vocab of 4096 allows
  at most 3839 merges.

## Library BPE and the bridge

The real run uses Hugging Face `tokenizers` (`tokenizer/train.py`). It is the same idea as our
scratch version, implemented in Rust so it can chew through millions of stories.

- **Byte-level.** Like ours, it starts from the 256 byte values, so nothing is ever "unknown".
- **Train split only.** The tokenizer is trained on training stories (capped at 50,000), never on
  validation stories. Otherwise validation text would leak into the vocabulary and make validation
  loss look better than it deserves. A test plants a rare marker word in a validation story and
  checks it never enters the vocab.
- **Same split everywhere.** `data/stories.py` owns the seeded shuffle that decides which
  stories are validation, so tokenizer training and token-file preparation always agree.
- **The bridge.** `tests/test_tokenizer_train.py::test_bridge_scratch_vs_library` trains both
  tokenizers on the same sample and checks that each round-trips the text exactly and uses fewer
  tokens than raw bytes. We do not claim they learn identical merges; the library's pre-splitting
  and tie-breaking differ from ours.

Run it: `grim-reaper tokenizer train configs/debug.yaml` (after `data download`).

## The data pipeline

```
grim-reaper data download   configs/debug.yaml   # data/tinystories/stories.txt
grim-reaper tokenizer train configs/debug.yaml   # data/tinystories/tokenizer.json
grim-reaper data prepare    configs/debug.yaml   # train.bin, val.bin, meta.json
```

1. **Download** fetches `TinyStories-train.txt` (pinned to a dataset revision) and saves it as
   `stories.txt`. Re-running skips it; `--force` downloads again.
2. **Split.** Stories are separated by `<|endoftext|>`. A seeded shuffle (`train.seed`) puts the
   last `val_fraction` (at least one story) into validation. Same config, same split.
3. **Tokenize.** Stories are encoded in batches and written as raw `uint16` ids with an
   end-of-text id after every story, so the model can learn where stories end.
4. **Provenance.** `meta.json` records the dataset revision, hashes of the text and tokenizer,
   the seed, story and token counts, and library versions. `prepare` compares these on re-run
   and skips work that is already up to date.

`debug.yaml` and `small.yaml` share `data/tinystories/`, so preparing one replaces the other's
files. Switching configs needs two steps: `tokenizer train` again, then `data prepare`.

`tokenizer train` also writes `tokenizer.meta.json` next to the tokenizer, recording the source
text hash, seed, `max_stories`, `val_fraction` and vocab size it was built for. `prepare` refuses a
tokenizer whose record does not match the current config. Without this, running `prepare` for a
config with a different split would reuse a vocabulary that had already seen some of its
validation stories.

If something is wrong, you get a message and not a traceback: a tokenizer whose vocab size does
not match `model.vocab_size`, a missing tokenizer, a failed download, or too few stories to split.

## Measured numbers

Apple M2, 8 GB RAM, Python 3.15.2, `tokenizers` 0.23.3. Measured with `/usr/bin/time -l`; one run each.

| Config | Stories (train / val) | Tokens (train / val) | `tokenizer train` | `data prepare` |
|---|---|---|---|---|
| `debug.yaml` (first 20,000 stories) | 19,800 / 200 | 4,444,359 / 45,781 | 0.8 s, 117 MB peak | 215 MB peak |
| `small.yaml` (full dataset) | 2,098,294 / 21,195 (230 empty dropped) | 472,556,147 / 4,775,221 | 6 s, 2.5 GB peak | 112 s, 2.7 GB peak |

The peak memory is dominated by holding every story's text in memory at once, not by tokenizing.
That is fine at this scale; a much larger dataset would need streaming splits.
