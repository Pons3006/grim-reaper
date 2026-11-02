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
