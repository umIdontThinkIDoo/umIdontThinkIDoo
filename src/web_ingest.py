"""Mirror web-derived text into the existing Library RAG (ChromaDB).

WHY THIS EXISTS
---------------
Two flows produce fresh, externally-fetched knowledge that the model otherwise
sees only for a single turn and then forgets:

  1. Chat web search (``comprehensive_web_search``) — fetched page content is
     pasted into the prompt for one answer, then discarded.
  2. Deep research — the final report lives only in the research JSON / DB.

The user wants that knowledge to become *immediately retrievable* in normal
chat. Rather than stand up a separate "fast" store, we reuse the SAME VectorRAG
that the Library/personal-docs use, with the SAME metadata shape, so retrieval,
ownership filtering, and the Library UI all stay uniform. This module is the one
place that knows how to turn web text into Library chunks.

Design rules:
- Best-effort and non-fatal. If ChromaDB is down or embedding fails, callers
  must not break; we log and return 0.
- Owner-scoped. Web content is written under the requesting user's ``owner`` so
  it shows up in their owner-filtered ``search(query, owner=...)`` (the chat RAG
  read path), exactly like their uploaded docs.
- Idempotent. VectorRAG derives doc ids from (owner, text), so re-ingesting the
  same page/report is a no-op rather than a duplicate.
- Off the request path. ``ingest_*_async`` fire on a daemon thread so a slow
  embed never delays the chat response or the research callback.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap how much of any single source we embed. Web pages and research reports can
# be huge; the excerpt budget keeps the Library from ballooning while still
# capturing the substance.
_MAX_CHARS_PER_SOURCE = 24_000


def _rag():
    """The shared VectorRAG instance, or None if ChromaDB isn't available."""
    try:
        from src.rag_singleton import get_rag_manager
        return get_rag_manager()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("web_ingest: RAG unavailable: %s", e)
        return None


def _chunks(rag, text: str) -> List[str]:
    """Chunk text the same way personal-doc ingest does, for uniformity."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) > _MAX_CHARS_PER_SOURCE:
        text = text[:_MAX_CHARS_PER_SOURCE]
    try:
        return rag._split_into_chunks(text)
    except Exception:
        # If the private chunker ever moves, fall back to one chunk rather than
        # losing the content entirely.
        return [text]


def ingest_text(
    text: str,
    *,
    owner: str,
    title: str,
    source: str,
    origin: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> int:
    """Embed one text blob into the Library RAG. Returns chunks written (0 on no-op).

    Metadata mirrors ``index_personal_documents`` (``source``/``filename``/
    ``type``/``owner``/``chunk_id``) so chat retrieval and the Library treat web
    content like any other document, plus ``origin`` and ``url`` so it can be
    told apart and cited.
    """
    rag = _rag()
    if rag is None:
        return 0
    chunks = _chunks(rag, text)
    if not chunks:
        return 0

    base: Dict[str, Any] = {
        "source": source,
        "filename": title or source,
        "title": title or source,
        "url": source,
        "type": "web",
        "origin": origin,  # "web_search" | "deep_research"
        "ingested_at": int(time.time()),
    }
    if owner:
        base["owner"] = owner
    if extra_meta:
        base.update(extra_meta)

    docs = [(chunk, {**base, "chunk_id": i}) for i, chunk in enumerate(chunks)]
    try:
        res = rag.add_documents_batch(docs)
    except Exception as e:
        logger.warning("web_ingest: add_documents_batch failed for %s: %s", source, e)
        return 0
    if not res.get("success"):
        logger.debug("web_ingest: batch not stored for %s: %s", source, res.get("message"))
        return 0
    added = int(res.get("added_count", 0))
    # Best-effort knowledge-graph enrichment, keyed to the same deterministic doc
    # ids the RAG store derives, so web-search / deep-research pages contribute
    # entities + relations too. Async + guarded: never blocks or faults ingest.
    try:
        from src import knowledge_graph as kg
        from src.rag_vector import _generate_doc_id
        kg.enrich_async([(_generate_doc_id(chunk, owner), chunk) for chunk, _m in docs], owner)
    except Exception:
        pass
    return added


def ingest_web_pages(query: str, pages: List[Dict[str, Any]], owner: str) -> int:
    """Embed fetched web pages (``comprehensive_web_search`` content dicts).

    ``pages`` items carry at least ``url``, ``title`` and ``content`` (the shape
    produced by ``fetch_webpage_content``). Returns total chunks written.
    """
    if not pages:
        return 0
    total = 0
    for p in pages:
        url = (p.get("url") or "").strip()
        content = p.get("content") or ""
        if not url or not content.strip():
            continue
        total += ingest_text(
            content,
            owner=owner,
            title=(p.get("title") or url),
            source=url,
            origin="web_search",
            extra_meta={"query": query} if query else None,
        )
    if total:
        logger.info("web_ingest: mirrored %d web chunk(s) into Library for owner=%s", total, owner or "-")
    return total


def ingest_web_pages_async(query: str, pages: List[Dict[str, Any]], owner: str) -> None:
    """Fire-and-forget ``ingest_web_pages`` on a daemon thread (never blocks)."""
    if not pages:
        return

    def _run() -> None:
        try:
            ingest_web_pages(query, pages, owner)
        except Exception as e:  # never let ingest crash a chat turn
            logger.warning("web_ingest: async web-page ingest failed: %s", e)

    threading.Thread(target=_run, name="web-ingest", daemon=True).start()


def ingest_research_report(
    *,
    query: str,
    report: str,
    owner: str,
    session_id: str = "",
) -> int:
    """Save a finished deep-research report into the Library RAG. Returns chunks."""
    if not (report or "").strip():
        return 0
    title = f"Research: {query.strip()[:120]}" if query else "Research report"
    source = f"research://{session_id}" if session_id else f"research://{abs(hash(query)) & 0xffffffff:x}"
    n = ingest_text(
        report,
        owner=owner,
        title=title,
        source=source,
        origin="deep_research",
        extra_meta={"query": query, "session_id": session_id} if query else {"session_id": session_id},
    )
    if n:
        logger.info("web_ingest: saved research report to Library (%d chunk(s)) for owner=%s", n, owner or "-")
    return n


def ingest_research_report_async(*, query: str, report: str, owner: str, session_id: str = "") -> None:
    """Fire-and-forget ``ingest_research_report`` on a daemon thread."""
    if not (report or "").strip():
        return

    def _run() -> None:
        try:
            ingest_research_report(query=query, report=report, owner=owner, session_id=session_id)
        except Exception as e:
            logger.warning("web_ingest: async research ingest failed: %s", e)

    threading.Thread(target=_run, name="research-ingest", daemon=True).start()
