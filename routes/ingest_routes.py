#!/usr/bin/env python3
"""
Bulk RAG ingest routes — accepts files/folders, extracts text,
chunks and embeds into the existing ChromaDB RAG system.

Routes (all under /api/ingest):
  POST   /upload        — multipart upload of one or more files
  GET    /progress      — current job state (polling)
  GET    /stats         — RAG collection stats
  POST   /stop          — graceful stop at the next book boundary
  POST   /clear-state   — reset progress state (not data)
  DELETE /document      — remove a previously ingested source
"""

import asyncio
import io
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.auth_helpers import require_user

logger = logging.getLogger(__name__)


# ── Module-level state ────────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "job_id": None,
    "files": [],
    "total_files": 0,
    "done_files": 0,
    "total_chunks": 0,
    "running": False,
    # Set by POST /stop. _run_ingest_job checks it between files and halts at the
    # next book boundary, leaving in-flight work intact and marking the rest
    # "deferred" so the UI can show what's left to re-ingest.
    "stop_requested": False,
}
_state_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)


# ── Text extraction helpers ───────────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(pages)
    except ImportError:
        raise RuntimeError("pypdf not installed — run: pip install pypdf")


def _extract_epub(path: Path) -> str:
    try:
        import ebooklib
        from ebooklib import epub
        from html.parser import HTMLParser

        class _Strip(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts: List[str] = []
            def handle_data(self, data):
                self._parts.append(data)
            def get_text(self):
                return " ".join(self._parts)

        book = epub.read_epub(str(path))
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            p = _Strip()
            p.feed(item.get_content().decode("utf-8", errors="ignore"))
            parts.append(p.get_text())
        return "\n\n".join(parts)
    except ImportError:
        raise RuntimeError("ebooklib not installed — run: pip install ebooklib")


def _extract_docx(path: Path) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise RuntimeError("python-docx not installed — run: pip install python-docx")


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".epub":
        return _extract_epub(path)
    if suffix in (".docx",):
        return _extract_docx(path)
    return path.read_text(errors="ignore")


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent)
        if sent_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            for i in range(0, sent_len, chunk_size - overlap):
                chunk = sent[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            continue
        if current_len + sent_len + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_chars = 0
            overlap_sents: List[str] = []
            for s in reversed(current):
                if overlap_chars + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_chars += len(s)
            current = overlap_sents
            current_len = overlap_chars
        current.append(sent)
        current_len += sent_len + 1
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.strip()) > 30]


# ── Library mirroring ─────────────────────────────────────────────────────────

def _add_pdf_to_library(name: str, file_bytes: bytes, owner: str, upload_handler) -> Optional[str]:
    """Persist an ingested PDF into the Library so it's viewable/readable and
    confirms what was chunked. Reuses the exact same path as
    /api/documents/import-pdf: save the bytes via upload_handler, then create a
    plain `pdf_source`-marked Document the PDF viewer can render.

    Idempotent on re-ingest: upload_handler.save_upload dedups identical bytes by
    hash (returns the existing upload id), and we skip creating a second Document
    when one already points at that upload. Best-effort — a failure here is
    logged and never fails the ingest (the chunks are what matter).
    """
    try:
        from starlette.datastructures import Headers, UploadFile as StarletteUploadFile
        from src.pdf_form_doc import create_plain_pdf_document
        from core.database import SessionLocal, Document
    except Exception as exc:
        logger.warning("Library mirror unavailable (import failed): %s", exc)
        return None

    try:
        upload = StarletteUploadFile(
            file=io.BytesIO(file_bytes),
            filename=name,
            headers=Headers({"content-type": "application/pdf"}),
        )
        # client_ip="ingest" is a synthetic source label for rate-limit bookkeeping.
        meta = upload_handler.save_upload(upload, "ingest", owner=owner)
        upload_id = meta["id"]

        # Dedup: a Document already pointing at this upload means we (or a prior
        # import) already mirrored it — don't create a duplicate library entry.
        db = SessionLocal()
        try:
            existing = db.query(Document).filter(
                Document.current_content.like(f'%upload_id="{upload_id}"%')
            ).first()
            if existing:
                return existing.id
        finally:
            db.close()

        title = os.path.splitext(name)[0]
        doc_id = create_plain_pdf_document(
            session_id=None, upload_id=upload_id, title=title, body_text=None
        )
        # A session-less import leaves owner NULL, which the Library's owner
        # filter then hides — stamp the ingesting user so the doc shows up.
        if doc_id and owner:
            db = SessionLocal()
            try:
                doc = db.query(Document).filter(Document.id == doc_id).first()
                if doc and not doc.owner:
                    doc.owner = owner
                    db.commit()
            finally:
                db.close()
        return doc_id
    except Exception as exc:
        logger.warning("Library mirror failed for %s: %s", name, exc)
        return None


# ── Background processing ─────────────────────────────────────────────────────

def _process_file(file_entry: Dict, file_bytes: bytes, rag_manager, owner: str, upload_handler=None):
    name = file_entry["name"]
    try:
        import tempfile
        suffix = Path(name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        with _state_lock:
            file_entry["status"] = "extracting"
            file_entry["progress"] = 10

        text = _extract_text(tmp_path)
        tmp_path.unlink(missing_ok=True)

        with _state_lock:
            file_entry["status"] = "chunking"
            file_entry["progress"] = 30

        chunks = _chunk_text(text)

        with _state_lock:
            file_entry["status"] = "embedding"
            file_entry["progress"] = 50
            file_entry["chunks"] = len(chunks)

        if chunks and rag_manager:
            # add_documents_batch expects (text, metadata) pairs and derives its
            # own deterministic doc ids from text+owner — passing a third element
            # raises "too many values to unpack" and fails the whole batch.
            docs = []
            for i, chunk in enumerate(chunks):
                meta = {
                    "source": name,
                    "owner": owner,
                    "chunk_id": i,
                    "ingest_source": "cookbook",
                }
                docs.append((chunk, meta))

            total = len(docs)
            batch_size = 20
            for batch_start in range(0, total, batch_size):
                batch = docs[batch_start:batch_start + batch_size]
                if hasattr(rag_manager, "add_documents_batch"):
                    rag_manager.add_documents_batch(batch)
                else:
                    for text_chunk, meta in batch:
                        rag_manager.add_document(text_chunk, meta)
                pct = 50 + int(50 * (batch_start + len(batch)) / total)
                with _state_lock:
                    file_entry["progress"] = pct

            # Best-effort knowledge-graph enrichment. Runs in a background
            # thread keyed to the same deterministic doc ids the RAG store
            # derives, so it never blocks this upload finishing and a failure
            # can't fault the ingest.
            try:
                from src import knowledge_graph as kg
                from src.rag_vector import _generate_doc_id

                kg.enrich_async(
                    [(_generate_doc_id(text_chunk, owner), text_chunk)
                     for text_chunk, _meta in docs],
                    owner,
                )
            except Exception:
                pass

        # Mirror PDFs into the Library so the user can confirm what was chunked
        # and actually read them. Only for PDFs (the viewer renders PDF pages),
        # only when chunks were produced, and best-effort (never blocks "done").
        if upload_handler and name.lower().endswith(".pdf") and chunks:
            with _state_lock:
                file_entry["status"] = "library"
                file_entry["progress"] = 99
            _add_pdf_to_library(name, file_bytes, owner, upload_handler)

        with _state_lock:
            file_entry["status"] = "done"
            file_entry["progress"] = 100
            _state["done_files"] += 1
            _state["total_chunks"] += len(chunks)

    except Exception as exc:
        with _state_lock:
            file_entry["status"] = "error"
            file_entry["error"] = str(exc)
            file_entry["progress"] = 0
            _state["done_files"] += 1

    with _state_lock:
        if _state["done_files"] >= _state["total_files"]:
            _state["running"] = False


def _run_ingest_job(files_data: List[tuple], rag_manager, owner: str, upload_handler=None):
    """files_data: list of (file_entry_dict, bytes)"""
    from src import ingest_gate
    with _state_lock:
        _state["running"] = True
    # Open the shared gate so chat pauses while we saturate the GPU with
    # embedding + KG extraction. try/finally guarantees the gate closes even if
    # a file blows up, so chat can never get stuck locked.
    ingest_gate.activate("upload", "Ingesting your library", total=len(files_data))

    try:
        for idx, (file_entry, file_bytes) in enumerate(files_data):
            # Graceful stop ("Interrupt ASAP"): honour a stop request — from the
            # cookbook UI (_state) or the global gate banner — at the book
            # boundary. Finish the files already done, never abandon one
            # mid-embed. Whatever hasn't started is marked "deferred" so the UI
            # lists what to re-ingest later (re-selecting the folder is
            # idempotent: done books are skipped).
            with _state_lock:
                stop = _state["stop_requested"]
            if stop or ingest_gate.stop_requested():
                with _state_lock:
                    for entry, _ in files_data:
                        if entry.get("status") == "queued":
                            entry["status"] = "deferred"
                            _state["done_files"] += 1
                    _state["running"] = False
                    _state["stop_requested"] = False
                return
            ingest_gate.update(done=idx, current=file_entry.get("name", ""))
            _process_file(file_entry, file_bytes, rag_manager, owner, upload_handler)
            ingest_gate.update(done=idx + 1)
    finally:
        ingest_gate.deactivate()


# ── Knowledge-graph backfill ───────────────────────────────────────────────────
#
# One-shot enrichment of chunks that were embedded BEFORE the utility model was
# configured (so they never got entities/relations). Reuses the same shared gate
# as upload ingest: chat pauses, the GPU runs flat-out on extraction, and the
# "Interrupt ASAP" button halts it at the next batch boundary. Idempotent +
# resumable — already-enriched doc ids are skipped, so re-running after an
# interrupt only picks up what's left.

def _owner_chunks(rag_manager, owner: str) -> List[tuple]:
    """All (doc_id, text) pairs for ``owner`` across every Chroma collection.

    Uses the stored doc ids directly (not regenerated) so KG neighbour
    resolution lines up with what query-time expansion reads back.
    """
    out: List[tuple] = []
    seen = set()
    try:
        collections = rag_manager._collections_for_delete()
    except Exception:
        collections = rag_manager._active_collections() if hasattr(rag_manager, "_active_collections") else []
    for _lane, col in collections:
        if col is None:
            continue
        try:
            got = col.get(where={"owner": owner}, include=["documents"])
        except Exception as exc:
            logger.warning("kg backfill: chunk fetch failed for lane: %s", exc)
            continue
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        for i, did in enumerate(ids):
            if did in seen:
                continue
            txt = docs[i] if i < len(docs) else ""
            if txt and txt.strip():
                seen.add(did)
                out.append((did, txt))
    return out


def _enrich_batch_concurrent(batch: List[tuple], owner: str, kg, workers: int = 4) -> None:
    """Extract + store a batch in parallel to keep the LLM/GPU busy.

    hermes3:8b via Ollama serves concurrent requests; a small pool pushes
    throughput while leaving VRAM headroom. Each item is independent and
    best-effort — one bad chunk never sinks the batch.
    """
    def _one(item):
        did, txt = item
        try:
            ents, rels = kg.extract(txt, owner)
            if ents:
                kg.store(owner, did, ents, rels)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(_one, batch))


def _run_kg_backfill(owner: str, workers: int = 4, batch_size: int = 16) -> None:
    from src import ingest_gate
    from src import knowledge_graph as kg

    ingest_gate.activate("kg_backfill", "Backfilling knowledge graph", total=0)
    try:
        rag_manager = _get_rag()
        if rag_manager is None or not kg.is_enabled():
            logger.warning("kg backfill: RAG unavailable or KG disabled — nothing to do")
            return

        pairs = _owner_chunks(rag_manager, owner)
        already = kg.existing_doc_ids(owner)
        todo = [(did, txt) for did, txt in pairs if did not in already]
        total = len(todo)
        ingest_gate.update(total=total, done=0, current=f"0 / {total} chunks")
        logger.info("kg backfill: owner=%s total_chunks=%d already=%d todo=%d",
                    owner or "-", len(pairs), len(already), total)

        done = 0
        for start in range(0, total, batch_size):
            if ingest_gate.stop_requested():
                logger.info("kg backfill: interrupted at %d/%d", done, total)
                break
            batch = todo[start:start + batch_size]
            _enrich_batch_concurrent(batch, owner, kg, workers=workers)
            done += len(batch)
            ingest_gate.update(done=done, current=f"{done} / {total} chunks")
        logger.info("kg backfill: finished — enriched up to %d/%d chunk(s) for %s",
                    done, total, owner or "-")
    except Exception as exc:
        logger.error("kg backfill failed: %s", exc)
    finally:
        ingest_gate.deactivate()


# ── Route factory ─────────────────────────────────────────────────────────────

def _get_rag():
    """Get rag_manager at request time so we're not bound to startup state."""
    try:
        from src.rag_singleton import get_rag_manager
        return get_rag_manager()
    except Exception:
        return None


def setup_ingest_routes(rag_manager=None, rag_available: bool = False, upload_handler=None):
    router = APIRouter(prefix="/api/ingest", tags=["ingest"])

    ALLOWED = {".pdf", ".epub", ".txt", ".md", ".docx", ".rst", ".csv"}
    MAX_FILE_MB = 200

    @router.post("/upload")
    async def ingest_upload(
        request: Request,
        files: List[UploadFile] = File(...),
        owner: str = Depends(require_user),
    ):
        rm = rag_manager or _get_rag()
        if rm is None:
            raise HTTPException(503, "RAG system unavailable — is ChromaDB running?")

        # `owner` comes from require_user (the canonical request.state.current_user
        # the auth middleware stamps), so ingested chunks are attributed to the
        # uploading user — not silently to "admin" as a stale request.state.user
        # read did. Empty string is the documented single-user/anonymous owner.

        if _state["running"]:
            raise HTTPException(409, "Ingest job already running — wait for it to finish")

        entries = []
        files_data = []

        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in ALLOWED:
                entries.append({
                    "name": upload.filename,
                    "size": 0,
                    "status": "error",
                    "progress": 0,
                    "chunks": 0,
                    "error": f"Unsupported type '{suffix}' — allowed: {', '.join(sorted(ALLOWED))}",
                })
                continue

            data = await upload.read()
            size_mb = len(data) / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                entries.append({
                    "name": upload.filename,
                    "size": len(data),
                    "status": "error",
                    "progress": 0,
                    "chunks": 0,
                    "error": f"File too large ({size_mb:.1f} MB > {MAX_FILE_MB} MB limit)",
                })
                continue

            entry = {
                "name": upload.filename,
                "size": len(data),
                "status": "queued",
                "progress": 0,
                "chunks": 0,
                "error": None,
            }
            entries.append(entry)
            files_data.append((entry, data))

        with _state_lock:
            _state["job_id"] = str(uuid.uuid4())[:8]
            _state["files"] = entries
            _state["total_files"] = len(files_data)
            _state["done_files"] = 0
            _state["total_chunks"] = 0
            _state["running"] = bool(files_data)
            _state["stop_requested"] = False

        if files_data:
            _executor.submit(_run_ingest_job, files_data, rm, owner, upload_handler)

        return {"job_id": _state["job_id"], "queued": len(files_data), "skipped": len(entries) - len(files_data)}

    @router.get("/progress")
    async def ingest_progress():
        with _state_lock:
            return dict(_state)

    @router.get("/stats")
    async def ingest_stats():
        rm = rag_manager or _get_rag()
        if rm is None:
            return {"available": False}
        try:
            stats = rm.get_stats() if hasattr(rm, "get_stats") else {}
            return {"available": True, **stats}
        except Exception as e:
            return {"available": False, "error": str(e)}

    @router.post("/stop")
    async def ingest_stop(owner: str = Depends(require_user)):
        """Request a graceful stop ("Interrupt ASAP"). The worker finishes the
        unit it's on (current book / current batch), then halts at the next
        boundary, frees the GPU, and unlocks chat. Works for BOTH the upload
        ingest and the KG backfill (the shared gate carries the stop flag).
        Returns deferred upload files (if any) so the UI can show what's left;
        re-selecting the same folder / re-running the backfill is idempotent."""
        from src import ingest_gate
        gate_stopped = ingest_gate.request_stop()
        with _state_lock:
            running = _state["running"]
            if running:
                _state["stop_requested"] = True
            deferred = [f["name"] for f in _state["files"]
                        if f.get("status") == "queued"] if running else []
        if not (running or gate_stopped):
            return {"ok": True, "running": False, "deferred": []}
        return {"ok": True, "stopping": True, "deferred": deferred, "deferred_count": len(deferred)}

    @router.get("/gate")
    async def ingest_gate_status():
        """Lightweight snapshot of the shared ingest gate for the global chat
        lock banner (cheap to poll: pure in-memory read)."""
        from src import ingest_gate
        return ingest_gate.snapshot()

    @router.post("/backfill-kg")
    async def ingest_backfill_kg(owner: str = Depends(require_user)):
        """Enrich every already-embedded chunk for this user into the knowledge
        graph. Runs in the background behind the shared gate (chat pauses,
        GPU saturates, Interrupt ASAP halts it). Idempotent + resumable."""
        from src import ingest_gate
        if ingest_gate.is_active() or _state["running"]:
            raise HTTPException(409, "An ingest or backfill is already running")
        _executor.submit(_run_kg_backfill, owner)
        return {"ok": True, "started": True}

    @router.post("/clear-state")
    async def ingest_clear_state():
        if _state["running"]:
            raise HTTPException(409, "Job still running")
        with _state_lock:
            _state.update({"job_id": None, "files": [], "total_files": 0,
                           "done_files": 0, "total_chunks": 0, "running": False,
                           "stop_requested": False})
        return {"ok": True}

    return router
