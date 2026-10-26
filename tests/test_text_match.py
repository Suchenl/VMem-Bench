"""Offline unit checks for text<->text helpers (Q2). No GPU/service: a deterministic bag-of-words
``embed_fn`` stands in for Qwen3-Embedding-4B so cosine reflects lexical/semantic overlap.

Run: cd benchmarks/MemStrata && PYTHONPATH=src <python> tests/test_text_match.py
"""

from __future__ import annotations

import math

from vmem_bench.annotation.pipeline_track_first.text_match import semantic_dedup, prompt_completeness

_VOCAB = ["rabbit", "bunny", "grey", "big", "apple", "red", "fruit", "butterfly", "oak", "tree"]
# synonyms collapse to a shared token so "bunny" ~ "rabbit" semantically.
_SYN = {"bunny": "rabbit", "fruit": "apple"}


def _embed(texts: list[list[str]] | list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        toks = [_SYN.get(w, w) for w in t.lower().replace(".", " ").replace(",", " ").split()]
        v = [float(sum(1 for w in toks if w == voc)) for voc in _VOCAB]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def test_semantic_dedup_merges_synonymous_names() -> None:
    entries = [
        {"name": "grey rabbit", "kind": "character", "description": "a big grey rabbit",
         "grounding_phrase": "grey rabbit", "static_attributes": {}},
        {"name": "the bunny", "kind": "character", "description": "big grey bunny",
         "grounding_phrase": "the bunny", "static_attributes": {}},
        {"name": "red apple", "kind": "prop", "description": "a red apple fruit",
         "grounding_phrase": "red apple", "static_attributes": {}},
    ]
    out = semantic_dedup(entries, _embed, threshold=0.7)
    names = sorted(e["name"] for e in out)
    assert len(out) == 2, out               # rabbit+bunny merged, apple separate
    assert names == ["grey rabbit", "red apple"]   # first-seen wins
    assert out[0]["grounding_phrase"] == "grey rabbit"


def test_semantic_dedup_static_conflict_blocks_merge() -> None:
    entries = [
        {"name": "critter", "kind": "character", "description": "grey rabbit",
         "grounding_phrase": "critter", "static_attributes": {"species": "rabbit"}},
        {"name": "critter2", "kind": "character", "description": "grey rabbit",
         "grounding_phrase": "critter2", "static_attributes": {"species": "bird"}},
    ]
    out = semantic_dedup(entries, _embed, threshold=0.5)
    assert len(out) == 2   # identical text but species conflict -> stays split


def test_semantic_dedup_different_kind_not_merged() -> None:
    entries = [
        {"name": "apple", "kind": "prop", "description": "red apple", "static_attributes": {}},
        {"name": "apple sign", "kind": "location", "description": "red apple", "static_attributes": {}},
    ]
    assert len(semantic_dedup(entries, _embed, threshold=0.5)) == 2


def test_prompt_completeness_flags_missing_entity() -> None:
    items = [("char_bunny", "grey rabbit. a big grey rabbit"),
             ("prop_apple", "red apple. a red apple fruit")]
    prompt = "The big grey rabbit hops across the meadow"   # apple not mentioned
    res = prompt_completeness(items, prompt, _embed, threshold=0.3)
    assert "prop_apple" in res["flagged"]
    assert "char_bunny" not in res["flagged"]
    assert res["scores"]["char_bunny"] > res["scores"]["prop_apple"]


def test_prompt_completeness_empty_inputs() -> None:
    assert prompt_completeness([], "x", _embed)["flagged"] == []
    assert prompt_completeness([("a", "b")], "", _embed)["flagged"] == []


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")


if __name__ == "__main__":
    _run_all()
