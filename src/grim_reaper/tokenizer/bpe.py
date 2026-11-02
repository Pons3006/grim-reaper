"""A from-scratch byte-level BPE tokenizer, written to be read.

Byte-pair encoding (BPE) builds a vocabulary by repeatedly merging the most frequent
neighbouring pair of tokens into a new token:

1. Start from raw bytes. Ids 0-255 are the 256 possible byte values, so *any* text can be
   encoded (no "unknown word" problem), including emoji and other languages.
2. Count how often each adjacent pair of ids occurs in the training text.
3. Give the most frequent pair a brand-new id and replace every occurrence with it.
4. Repeat until the vocabulary is the size we asked for.

The ordered list of merges *is* the tokenizer. To encode new text we replay the merges in the
order they were learned. This version is pure Python and meant for small corpora; the real
run uses Hugging Face `tokenizers`, which does the same thing in fast Rust.

Vocabulary layout: ids 0-255 bytes, then one id per merge, and the last id is the special
end-of-text token. So a vocab of size V allows at most V - 257 merges.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

SPECIAL_TOKEN = "<|endoftext|>"

# Words (with an optional leading space), runs of punctuation, and whitespace. Merges never
# cross these boundaries, so "cat sat" can never become one token spanning the space gap.
_PIECE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")
_SPECIAL_SPLIT = re.compile(f"({re.escape(SPECIAL_TOKEN)})")

Pair = tuple[int, int]
MergeCallback = Callable[[int, Pair, int, int], None]


def _pieces(text: str) -> list[str]:
    return _PIECE.findall(text)


def _merge_pair(ids: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    """Replace every adjacent occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, merges: list[Pair], vocab_size: int) -> None:
        if len(merges) > vocab_size - 257:
            raise ValueError(f"{len(merges)} merges do not fit in vocab_size={vocab_size}")
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = vocab_size
        self.special_id = vocab_size - 1
        self._rank = {pair: i for i, pair in enumerate(self.merges)}
        self._bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self._bytes[256 + i] = self._bytes[a] + self._bytes[b]
        self._bytes[self.special_id] = SPECIAL_TOKEN.encode()

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        on_merge: MergeCallback | None = None,
    ) -> BPETokenizer:
        """Learn merges from `texts`. `on_merge(step, pair, new_id, count)` is called per merge."""
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 (256 bytes + special token)")
        # Count each distinct piece once; pair counts are weighted by how often it occurs.
        freqs: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for part in _SPECIAL_SPLIT.split(text):
                if part == SPECIAL_TOKEN:
                    continue
                for piece in _pieces(part):
                    freqs[tuple(piece.encode())] += 1

        merges: list[Pair] = []
        while len(merges) < vocab_size - 257:
            counts: Counter[Pair] = Counter()
            for ids, freq in freqs.items():
                for pair in zip(ids, ids[1:], strict=False):
                    counts[pair] += freq
            if not counts:
                break
            # Most frequent pair; ties go to the smaller pair so training is deterministic.
            pair, count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if count < 2:
                break  # nothing repeats, so merging would just memorise the corpus
            new_id = 256 + len(merges)
            merges.append(pair)
            freqs = Counter({_merge_pair(ids, pair, new_id): f for ids, f in freqs.items()})
            if on_merge is not None:
                on_merge(len(merges), pair, new_id, count)
        return cls(merges, vocab_size)

    def _encode_piece(self, piece: str) -> list[int]:
        ids = list(piece.encode())
        while len(ids) >= 2:
            # Apply the earliest-learned merge that is present, exactly as training did.
            pair = min(
                zip(ids, ids[1:], strict=False), key=lambda p: self._rank.get(p, len(self.merges))
            )
            if pair not in self._rank:
                break
            ids = list(_merge_pair(tuple(ids), pair, 256 + self._rank[pair]))
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for part in _SPECIAL_SPLIT.split(text):
            if part == SPECIAL_TOKEN:
                out.append(self.special_id)
            else:
                for piece in _pieces(part):
                    out.extend(self._encode_piece(piece))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        data = {"vocab_size": self.vocab_size, "merges": [list(m) for m in self.merges]}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text())
        return cls([tuple(m) for m in data["merges"]], data["vocab_size"])
