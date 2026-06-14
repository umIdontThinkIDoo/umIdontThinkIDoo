"""Tests for src/knowledge_graph.py — the lightweight entity graph.

The LLM extractor is monkeypatched so tests stay fast/offline; the SQLite store
and the 1-hop expansion walk run for real against a temp DB.
"""
import pytest

from src import knowledge_graph as kg


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    kg.reset_for_tests(str(tmp_path / "kg.db"))
    # default-on regardless of any settings file on disk
    monkeypatch.setattr(kg, "_get_setting", lambda k, d: d)
    yield
    kg.reset_for_tests()


def test_store_and_one_hop_expansion():
    owner = "u1"
    # doc1 mentions A,B and relates them; doc2 mentions B,C; doc3 mentions C.
    kg.store(owner, "doc1", ["Alpha", "Beta"], [("Alpha", "Beta")])
    kg.store(owner, "doc2", ["Beta", "Gamma"], [("Beta", "Gamma")])
    kg.store(owner, "doc3", ["Gamma"], [])

    # Seeding from doc1 (entities Alpha, Beta): Beta links to Gamma (doc2/doc3),
    # and Beta itself anchors doc2.
    out = kg.neighbor_doc_ids(["doc1"], owner, limit=5)
    assert "doc2" in out
    assert "doc1" not in out  # seeds excluded


def test_expansion_respects_owner_scope():
    kg.store("u1", "doc1", ["Shared"], [])
    kg.store("u2", "doc2", ["Shared"], [])
    out = kg.neighbor_doc_ids(["doc1"], "u1", limit=5)
    assert "doc2" not in out  # different owner, never crosses


def test_expansion_excludes_provided_ids():
    kg.store("u1", "doc1", ["X", "Y"], [("X", "Y")])
    kg.store("u1", "doc2", ["Y"], [])
    out = kg.neighbor_doc_ids(["doc1"], "u1", exclude={"doc2"}, limit=5)
    assert "doc2" not in out


def test_expansion_limit_caps_results():
    kg.store("u1", "seed", ["Hub"], [])
    for i in range(10):
        kg.store("u1", f"d{i}", ["Hub"], [])
    out = kg.neighbor_doc_ids(["seed"], "u1", limit=3)
    assert len(out) == 3


def test_neighbor_empty_when_no_seeds():
    assert kg.neighbor_doc_ids([], "u1") == []


def test_neighbor_empty_when_unknown_seed():
    assert kg.neighbor_doc_ids(["nope"], "u1") == []


def test_parse_extraction_plain_json():
    ents, rels = kg._parse_extraction('{"entities": ["A", "B"], "relations": [["A", "B"]]}')
    assert ents == ["A", "B"]
    assert rels == [("A", "B")]


def test_parse_extraction_with_code_fence_and_chatter():
    raw = 'Sure!\n```json\n{"entities": ["A"], "relations": []}\n```\nDone.'
    ents, rels = kg._parse_extraction(raw)
    assert ents == ["A"]
    assert rels == []


def test_parse_extraction_garbage_is_safe():
    assert kg._parse_extraction("not json at all") == ([], [])
    assert kg._parse_extraction("") == ([], [])


def test_enrich_documents_uses_extractor(monkeypatch):
    monkeypatch.setattr(kg, "extract", lambda text, owner: (["Ent"], []))
    n = kg.enrich_documents([("docA", "some text")], "u1")
    assert n == 1
    # stored and findable as an anchor
    kg.store("u1", "docB", ["Ent"], [])
    out = kg.neighbor_doc_ids(["docA"], "u1", limit=5)
    assert "docB" in out


def test_enrich_documents_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(kg, "is_enabled", lambda: False)
    called = {"n": 0}

    def _boom(text, owner):
        called["n"] += 1
        return (["X"], [])

    monkeypatch.setattr(kg, "extract", _boom)
    assert kg.enrich_documents([("d", "t")], "u1") == 0
    assert called["n"] == 0


def test_extract_returns_empty_on_no_endpoint(monkeypatch):
    # No endpoint resolves → graceful empty, never raises.
    import src.endpoint_resolver as er
    monkeypatch.setattr(er, "resolve_endpoint", lambda *a, **k: (None, None, {}))
    assert kg.extract("text", "u1") == ([], [])
