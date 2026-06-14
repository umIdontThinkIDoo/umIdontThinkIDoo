"""Integration: VectorRAG.search applies cross-encoder reranking when enabled.

The lane/Chroma machinery is stubbed so the test stays fast and deterministic;
the focus is that search() (1) over-fetches a wider pool when reranking is on
and (2) returns the cross-encoder's order, while leaving the hybrid path
untouched when reranking is off.
"""
import pytest

import src.rag_vector as rag_vector
from src import reranker
from src.rag_vector import VectorRAG


class _FakeLane:
    name = "fastembed"

    def count(self):
        return 100


def _results(ids_docs):
    return {
        "ids": [[i for i, _ in ids_docs]],
        "documents": [[d for _, d in ids_docs]],
        "metadatas": [[{"source": i} for i, _ in ids_docs]],
        "distances": [[0.1 for _ in ids_docs]],
    }


@pytest.fixture
def store(monkeypatch):
    s = VectorRAG.__new__(VectorRAG)
    s._lanes = [_FakeLane()]
    s._healthy = True

    docs = [
        ("a", "the cat sat on the mat"),
        ("b", "quantum chromodynamics lattice"),
        ("c", "a feline rested on a rug"),
    ]
    captured = {}

    def _fake_query_lanes(lanes, query, n_results=None, where=None, include=None, raise_if_all_failed=False):
        # Record the per-lane n_results so we can assert over-fetch widening.
        captured["n_results"] = n_results(lanes[0]) if callable(n_results) else n_results
        yield lanes[0], _results(docs)

    monkeypatch.setattr(rag_vector, "query_lanes", _fake_query_lanes)
    monkeypatch.setattr(rag_vector, "lane_count", lambda lanes: 100)
    monkeypatch.setattr(rag_vector, "dedupe_results", lambda cands, limit=None, **kw: cands[:limit])
    reranker.reset_reranker_state()
    yield s, captured
    reranker.reset_reranker_state()


def test_search_returns_reranked_order(store, monkeypatch):
    s, captured = store
    monkeypatch.setattr(reranker, "is_enabled", lambda: True)
    # Cross-encoder prefers c, then a, then b.
    score_map = {"a feline rested on a rug": 0.9, "the cat sat on the mat": 0.5, "quantum chromodynamics lattice": 0.1}
    monkeypatch.setattr(reranker, "score", lambda q, docs: [score_map[d] for d in docs])

    out = s.search("cat on mat", k=2)
    assert [r["id"] for r in out] == ["c", "a"]
    # over-fetched a wider pool than k (k * overfetch, then *3 lane fan-out)
    assert captured["n_results"] > 2


def test_search_without_rerank_uses_hybrid_order(store, monkeypatch):
    s, captured = store
    monkeypatch.setattr(reranker, "is_enabled", lambda: False)
    monkeypatch.setattr(reranker, "score", lambda q, docs: pytest.fail("scored while disabled"))

    out = s.search("cat on mat", k=2)
    # "cat"/"mat" keyword overlap puts doc "a" first under the hybrid score.
    assert out[0]["id"] == "a"
    assert len(out) == 2
    # no widening when reranking is off
    assert captured["n_results"] <= max(2, 20)


def test_search_folds_in_kg_neighbors(store, monkeypatch):
    s, captured = store
    monkeypatch.setattr(reranker, "is_enabled", lambda: False)

    # A graph neighbour "z" that wasn't in the first-stage vector results.
    from src import knowledge_graph as kg
    monkeypatch.setattr(kg, "expansion_enabled", lambda: True)
    monkeypatch.setattr(kg, "expand_limit", lambda: 5)
    monkeypatch.setattr(kg, "neighbor_doc_ids", lambda seeds, owner, exclude=None, limit=5: ["z"])

    class _FakeColl:
        def get(self, ids, include=None):
            assert "z" in ids
            return {"ids": ["z"], "documents": ["a cat purred softly"],
                    "metadatas": [{"owner": "u1"}]}

    monkeypatch.setattr(s, "_active_collections", lambda: [("fastembed", _FakeColl())])

    out = s.search("cat", k=10, owner="u1")
    ids = [r["id"] for r in out]
    assert "z" in ids  # graph neighbour reached the result set
    assert any(r.get("kg_expanded") for r in out)
