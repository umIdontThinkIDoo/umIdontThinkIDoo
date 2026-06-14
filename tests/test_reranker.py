"""Tests for src/reranker.py — the cross-encoder reranking stage.

These never load a real ONNX model: the cross-encoder is monkeypatched via the
module-level `score()` function so the tests stay fast and deterministic while
still exercising the real ordering/fallback logic.
"""
import pytest

from src import reranker


@pytest.fixture(autouse=True)
def _reset_reranker():
    reranker.reset_reranker_state()
    yield
    reranker.reset_reranker_state()


def _candidates():
    return [
        {"id": "a", "document": "the cat sat on the mat", "similarity": 0.9},
        {"id": "b", "document": "quantum chromodynamics lattice", "similarity": 0.8},
        {"id": "c", "document": "a feline rested on a rug", "similarity": 0.7},
    ]


def test_rerank_reorders_by_cross_encoder(monkeypatch):
    # Cross-encoder loves candidate "c" most, then "a", then "b".
    monkeypatch.setattr(reranker, "is_enabled", lambda: True)
    monkeypatch.setattr(reranker, "score", lambda q, docs: [0.2, 0.1, 0.9])

    out = reranker.rerank("cat on mat", _candidates(), top_k=2)
    assert [c["id"] for c in out] == ["c", "a"]
    # winning candidate carries its rerank score
    assert out[0]["rerank_score"] == pytest.approx(0.9)
    # original first-stage fields are preserved
    assert out[0]["similarity"] == 0.7


def test_rerank_truncates_to_top_k(monkeypatch):
    monkeypatch.setattr(reranker, "is_enabled", lambda: True)
    monkeypatch.setattr(reranker, "score", lambda q, docs: [0.5, 0.4, 0.3])
    out = reranker.rerank("q", _candidates(), top_k=1)
    assert len(out) == 1
    assert out[0]["id"] == "a"


def test_disabled_returns_first_stage_order_untouched(monkeypatch):
    monkeypatch.setattr(reranker, "is_enabled", lambda: False)
    # score() must not even be called when disabled
    monkeypatch.setattr(reranker, "score", lambda q, docs: pytest.fail("scored while disabled"))
    out = reranker.rerank("q", _candidates(), top_k=2)
    assert [c["id"] for c in out] == ["a", "b"]
    assert "rerank_score" not in out[0]


def test_no_op_when_model_unavailable(monkeypatch):
    # Reranking enabled but the encoder can't load → graceful passthrough.
    monkeypatch.setattr(reranker, "is_enabled", lambda: True)
    monkeypatch.setattr(reranker, "score", lambda q, docs: None)
    out = reranker.rerank("q", _candidates(), top_k=2)
    assert [c["id"] for c in out] == ["a", "b"]
    assert "rerank_score" not in out[0]


def test_empty_candidates():
    assert reranker.rerank("q", [], top_k=5) == []


def test_text_fn_override(monkeypatch):
    monkeypatch.setattr(reranker, "is_enabled", lambda: True)
    seen = {}

    def _score(q, docs):
        seen["docs"] = list(docs)
        return [0.1, 0.9, 0.5]

    monkeypatch.setattr(reranker, "score", _score)
    cands = [{"id": "a", "body": "x"}, {"id": "b", "body": "y"}, {"id": "c", "body": "z"}]
    out = reranker.rerank("q", cands, top_k=3, text_fn=lambda c: c["body"])
    assert seen["docs"] == ["x", "y", "z"]
    assert [c["id"] for c in out] == ["b", "c", "a"]


def test_score_returns_none_without_encoder(monkeypatch):
    # _get_encoder unavailable → score() is None (not an exception).
    monkeypatch.setattr(reranker, "_get_encoder", lambda: None)
    assert reranker.score("q", ["a", "b"]) is None


def test_score_aligns_with_encoder(monkeypatch):
    class _FakeEncoder:
        def rerank(self, query, docs):
            # deterministic: score = length of each doc
            return [float(len(d)) for d in docs]

    monkeypatch.setattr(reranker, "_get_encoder", lambda: _FakeEncoder())
    scores = reranker.score("q", ["aa", "bbbb", "c"])
    assert scores == [2.0, 4.0, 1.0]


def test_overfetch_multiplier_default(monkeypatch):
    monkeypatch.setattr(reranker, "_get_setting", lambda k, d: d)
    assert reranker.overfetch_multiplier() == reranker.DEFAULT_OVERFETCH


def test_overfetch_multiplier_clamps_below_one(monkeypatch):
    monkeypatch.setattr(reranker, "_get_setting", lambda k, d: 0)
    assert reranker.overfetch_multiplier() == reranker.DEFAULT_OVERFETCH
