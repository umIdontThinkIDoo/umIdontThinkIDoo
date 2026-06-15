"""Regression tests for routes/ingest_routes.py (bulk RAG ingest).

Covers two real bugs:
  1. The batch path built 3-tuples (chunk, meta, doc_id) but
     VectorRAG.add_documents_batch unpacks 2-tuples (text, meta) and derives
     ids itself — the 3rd element raised "too many values to unpack" and failed
     every file.
  2. Owner was read from request.state.user (never set) and silently fell back
     to "admin"; it must come from the authenticated user via require_user.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.ingest_routes as ing
from routes.ingest_routes import _chunk_text, _process_file, setup_ingest_routes
from src.auth_helpers import require_user


class _RealisticRag:
    """Mimics VectorRAG.add_documents_batch's 2-tuple unpack so a 3-tuple
    regression would raise here exactly as it does in production."""
    def __init__(self):
        self.added = []

    def add_documents_batch(self, docs):
        for t, m in docs:          # would raise on a 3-tuple
            self.added.append((t, m))
        return {"success": True}


def test_process_file_feeds_two_tuples_and_completes():
    rag = _RealisticRag()
    entry = {"name": "doc.txt", "status": "queued", "progress": 0, "chunks": 0}
    # Reset shared module state the function mutates.
    ing._state.update({"done_files": 0, "total_files": 1, "total_chunks": 0, "running": True})

    text = ("Sentence one is here. Sentence two follows it. " * 80)
    _process_file(entry, text.encode(), rag, owner="alice")

    assert entry["status"] == "done", entry.get("error")
    assert entry["chunks"] > 0
    assert len(rag.added) == entry["chunks"]
    # Every chunk carried the uploading user's owner.
    assert all(meta["owner"] == "alice" for _, meta in rag.added)


def test_chunk_text_drops_trivial_and_splits_long():
    assert _chunk_text("   ") == []
    big = "word " * 5000
    chunks = _chunk_text(big, chunk_size=1000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_upload_attributes_owner_from_auth_not_admin(monkeypatch):
    """The owner handed to the ingest job must be the resolved current user
    (here the internal-tool identity), not a hardcoded "admin" fallback."""
    captured = {}

    def _fake_run(files_data, rag_manager, owner, upload_handler=None):
        captured["owner"] = owner

    monkeypatch.setattr(ing, "_run_ingest_job", _fake_run)
    # Make the executor run inline so the capture happens before we assert.
    monkeypatch.setattr(ing._executor, "submit", lambda fn, *a, **k: fn(*a, **k))
    ing._state["running"] = False

    app = FastAPI()
    app.include_router(setup_ingest_routes(rag_manager=_RealisticRag(), rag_available=True))
    # Stand in for the auth middleware: require_user resolves to a real user.
    app.dependency_overrides[require_user] = lambda: "alice"
    client = TestClient(app)

    r = client.post(
        "/api/ingest/upload",
        files=[("files", ("note.txt", b"hello world", "text/plain"))],
    )
    assert r.status_code == 200, r.text
    # Owner flows from require_user, not the old hardcoded "admin" fallback.
    assert captured["owner"] == "alice"
    assert captured["owner"] != "admin"
