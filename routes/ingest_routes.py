#!/usr/bin/env python3
"""
Bulk RAG ingest routes — accepts files/folders, extracts text,
chunks and embeds into the existing ChromaDB RAG system.

Routes (all under /api/ingest):
  POST   /upload        — multipart upload of one or more files
  GET    /progress      — current job state (polling)
  GET    /stats         — RAG collection stats
  POST   /clear-state   — reset progress state (not data)
  DELETE /document      — remove a previously ingested source
"""

import asyncio
import hashlib
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse


# ── Module-level state ────────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "job_id": None,
    "files": [],
    "total_files": 0,
    "done_files": 0,
    "total_chunks": 0,
    "running": False,
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


# ── Background processing ─────────────────────────────────────────────────────

def _process_file(file_entry: Dict, file_bytes: bytes, rag_manager, owner: str):
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
            source_id = hashlib.sha256(name.encode()).hexdigest()[:16]
            docs = []
            for i, chunk in enumerate(chunks):
                doc_id = f"{source_id}-{i}"
                meta = {
                    "source": name,
                    "owner": owner,
                    "chunk_id": i,
                    "ingest_source": "cookbook",
                }
                docs.append((chunk, meta, doc_id))

            total = len(docs)
            batch_size = 20
            for batch_start in range(0, total, batch_size):
                batch = docs[batch_start:batch_start + batch_size]
                if hasattr(rag_manager, "add_documents_batch"):
                    rag_manager.add_documents_batch(batch)
                else:
                    for text_chunk, meta, _ in batch:
                        rag_manager.add_document(text_chunk, meta)
                pct = 50 + int(50 * (batch_start + len(batch)) / total)
                with _state_lock:
                    file_entry["progress"] = pct

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


def _run_ingest_job(files_data: List[tuple], rag_manager, owner: str):
    """files_data: list of (file_entry_dict, bytes)"""
    with _state_lock:
        _state["running"] = True

    for file_entry, file_bytes in files_data:
        _process_file(file_entry, file_bytes, rag_manager, owner)


# ── Route factory ─────────────────────────────────────────────────────────────

def _get_rag():
    """Get rag_manager at request time so we're not bound to startup state."""
    try:
        from src.rag_singleton import get_rag_manager
        return get_rag_manager()
    except Exception:
        return None


def setup_ingest_routes(rag_manager=None, rag_available: bool = False):
    router = APIRouter(prefix="/api/ingest", tags=["ingest"])

    ALLOWED = {".pdf", ".epub", ".txt", ".md", ".docx", ".rst", ".csv"}
    MAX_FILE_MB = 200

    @router.post("/upload")
    async def ingest_upload(request: Request, files: List[UploadFile] = File(...)):
        rm = rag_manager or _get_rag()
        if rm is None:
            raise HTTPException(503, "RAG system unavailable — is ChromaDB running?")

        owner = getattr(request.state, "user", None) or "admin"

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

        if files_data:
            _executor.submit(_run_ingest_job, files_data, rm, owner)

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

    @router.post("/clear-state")
    async def ingest_clear_state():
        if _state["running"]:
            raise HTTPException(409, "Job still running")
        with _state_lock:
            _state.update({"job_id": None, "files": [], "total_files": 0,
                           "done_files": 0, "total_chunks": 0, "running": False})
        return {"ok": True}

    return router
