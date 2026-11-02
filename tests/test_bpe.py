from grim_reaper.tokenizer.bpe import SPECIAL_TOKEN, BPETokenizer

CORPUS = ["the cat sat on the mat. the cat ate the rat."] * 5


def test_base_ids_are_bytes():
    tok = BPETokenizer.train(["ab"], vocab_size=257 + 1)
    assert tok.merges == []
    assert tok.encode("a") == [97]


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer.train(["aaab aaab"], vocab_size=257 + 1)
    assert tok.merges == [(97, 97)]


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges


def test_round_trip_unicode():
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    for text in ["héllo 🌍", "naïve café", "日本語"]:
        assert tok.decode(tok.encode(text)) == text


def test_empty_string():
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_special_token_is_single_id():
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    text = f"a{SPECIAL_TOKEN}b"
    ids = tok.encode(text)
    assert ids.count(tok.special_id) == 1
    assert tok.decode(ids) == text


def test_merges_reduce_length():
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode())


def test_save_load_round_trip(tmp_path):
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    path = tmp_path / "bpe.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.encode(CORPUS[0]) == tok.encode(CORPUS[0])


def test_on_merge_called_per_merge():
    calls = []
    tok = BPETokenizer.train(
        CORPUS, vocab_size=300, on_merge=lambda step, pair, new_id, count: calls.append(step)
    )
    assert len(calls) == len(tok.merges)
    assert calls == list(range(1, len(calls) + 1))
