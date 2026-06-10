"""Training routes — SFT, DPO, CPT, QLoRA, and RL Loop via TRL.

Jobs run as supervised subprocesses inside tmux (same pattern as Cookbook serve).
Progress is streamed back via SSE from the job log file.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.middleware import require_admin
from core.platform_compat import find_bash, kill_process_tree, pid_alive, detached_popen_kwargs
from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

_TRAINING_STATE_PATH = Path(DATA_DIR) / "training_state.json"
_TRAINING_LOG_DIR = Path(DATA_DIR) / "training_logs"
_BOOK_UPLOAD_DIR = Path(DATA_DIR) / "training_books"

_ALLOWED_BOOK_EXTENSIONS = {".pdf", ".epub", ".txt", ".md", ".docx"}
_MAX_BOOK_SIZE_MB = 200


class TrainingRequest(BaseModel):
    job_type: str          # sft | dpo | cpt | qlora | rl_loop
    model_id: str
    dataset_path: Optional[str] = None
    book_filename: Optional[str] = None   # for rl_loop
    seed: Optional[int] = 42
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    learning_rate: Optional[float] = 2e-4
    epochs: Optional[int] = 3
    batch_size: Optional[int] = 2
    grad_accum: Optional[int] = 4
    base_adapter_path: Optional[str] = None  # for qlora merge
    output_name: Optional[str] = None


def _load_state() -> dict:
    if _TRAINING_STATE_PATH.exists():
        try:
            return json.loads(_TRAINING_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"jobs": []}


def _save_state(state: dict) -> None:
    _TRAINING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRAINING_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _job_log_path(job_id: str) -> Path:
    _TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _TRAINING_LOG_DIR / f"{job_id}.log"


def _find_job(state: dict, job_id: str) -> Optional[dict]:
    for job in state.get("jobs", []):
        if job.get("id") == job_id:
            return job
    return None


def _build_train_script(req: TrainingRequest, job_id: str, log_path: str) -> list[str]:
    """Build the python invocation for the requested training job type."""
    scripts_dir = Path(__file__).parent.parent / "scripts" / "training"
    script_map = {
        "sft":      scripts_dir / "sft_train.py",
        "dpo":      scripts_dir / "dpo_train.py",
        "cpt":      scripts_dir / "cpt_train.py",
        "qlora":    scripts_dir / "qlora_merge.py",
        "rl_loop":  scripts_dir / "rl_loop.py",
    }
    script = script_map.get(req.job_type)
    if not script:
        raise ValueError(f"Unknown job type: {req.job_type}")

    cmd = [
        "python", str(script),
        "--model-id", req.model_id,
        "--job-id", job_id,
        "--log-file", log_path,
        "--seed", str(req.seed or 42),
        "--lora-rank", str(req.lora_rank or 16),
        "--lora-alpha", str(req.lora_alpha or 32),
        "--lr", str(req.learning_rate or 2e-4),
        "--epochs", str(req.epochs or 3),
        "--batch-size", str(req.batch_size or 2),
        "--grad-accum", str(req.grad_accum or 4),
    ]
    if req.dataset_path:
        cmd += ["--dataset", req.dataset_path]
    if req.book_filename and req.job_type == "rl_loop":
        book_path = _BOOK_UPLOAD_DIR / req.book_filename
        cmd += ["--book", str(book_path)]
    if req.base_adapter_path and req.job_type == "qlora":
        cmd += ["--base-adapter", req.base_adapter_path]
    if req.output_name:
        cmd += ["--output-name", req.output_name]
    return cmd


def setup_training_routes() -> APIRouter:
    router = APIRouter(prefix="/api/training", tags=["training"])

    @router.get("/state")
    async def get_training_state(request: Request):
        require_admin(request)
        state = _load_state()
        # Refresh live pid status
        for job in state.get("jobs", []):
            if job.get("status") == "running":
                pid = job.get("pid")
                if pid and not pid_alive(pid):
                    # Check log for success/failure marker
                    log = _job_log_path(job["id"])
                    if log.exists():
                        text = log.read_text(errors="ignore")
                        job["status"] = "done" if "[TRAINING_OK]" in text else "error"
                    else:
                        job["status"] = "error"
        _save_state(state)
        return state

    @router.post("/start")
    async def start_training(request: Request, body: TrainingRequest):
        require_admin(request)
        state = _load_state()

        # Prevent concurrent jobs (GPU memory)
        for job in state.get("jobs", []):
            if job.get("status") == "running":
                raise HTTPException(409, "A training job is already running. Stop it first.")

        job_id = str(uuid.uuid4())[:8]
        log_path = str(_job_log_path(job_id))

        try:
            cmd = _build_train_script(body, job_id, log_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        bash = find_bash()
        if not bash:
            raise HTTPException(500, "bash not found — cannot launch training script")

        import subprocess
        kwargs = detached_popen_kwargs()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        except Exception as exc:
            raise HTTPException(500, f"Failed to start training: {exc}") from exc

        job = {
            "id": job_id,
            "type": body.job_type,
            "model_id": body.model_id,
            "status": "running",
            "pid": proc.pid,
            "config": body.model_dump(),
        }
        state.setdefault("jobs", []).insert(0, job)
        # Keep at most 50 jobs in history
        state["jobs"] = state["jobs"][:50]
        _save_state(state)
        logger.info("Training job %s started (pid %d): %s", job_id, proc.pid, body.job_type)
        return {"ok": True, "job_id": job_id}

    @router.post("/stop/{job_id}")
    async def stop_training(request: Request, job_id: str):
        require_admin(request)
        state = _load_state()
        job = _find_job(state, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        pid = job.get("pid")
        if pid and pid_alive(pid):
            kill_process_tree(pid)
        job["status"] = "stopped"
        _save_state(state)
        return {"ok": True}

    @router.get("/logs/{job_id}")
    async def stream_training_logs(request: Request, job_id: str, offset: int = 0):
        """SSE stream of training log lines starting at byte offset."""
        require_admin(request)
        log_path = _job_log_path(job_id)

        async def _generate():
            current_offset = offset
            while True:
                if await request.is_disconnected():
                    break
                if log_path.exists():
                    content = log_path.read_bytes()
                    if len(content) > current_offset:
                        chunk = content[current_offset:].decode("utf-8", errors="replace")
                        current_offset = len(content)
                        for line in chunk.splitlines():
                            yield f"data: {json.dumps({'line': line})}\n\n"
                # Stop streaming when job is no longer running
                state = _load_state()
                job = _find_job(state, job_id)
                if not job or job.get("status") not in ("running",):
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break
                await asyncio.sleep(0.75)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/upload-book")
    async def upload_book(request: Request, file: UploadFile = File(...)):
        """Upload a PDF/EPUB/TXT book for the RL Loop trainer."""
        require_admin(request)
        _BOOK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename or "").suffix.lower()
        if ext not in _ALLOWED_BOOK_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(_ALLOWED_BOOK_EXTENSIONS)}")

        # Sanitize filename
        safe_name = re.sub(r"[^\w\s\-.]", "_", file.filename or "book") + ""
        safe_name = safe_name[:120]
        dest = _BOOK_UPLOAD_DIR / safe_name

        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > _MAX_BOOK_SIZE_MB:
            raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max {_MAX_BOOK_SIZE_MB} MB.")

        dest.write_bytes(content)
        logger.info("Book uploaded: %s (%.1f MB)", safe_name, size_mb)
        return {"ok": True, "filename": safe_name, "size_mb": round(size_mb, 2)}

    @router.get("/books")
    async def list_books(request: Request):
        require_admin(request)
        _BOOK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        books = []
        for f in sorted(_BOOK_UPLOAD_DIR.iterdir()):
            if f.is_file() and f.suffix.lower() in _ALLOWED_BOOK_EXTENSIONS:
                books.append({"filename": f.name, "size_mb": round(f.stat().st_size / (1024*1024), 2)})
        return {"books": books}

    @router.delete("/books/{filename}")
    async def delete_book(request: Request, filename: str):
        require_admin(request)
        safe = re.sub(r"[^\w\s\-.]", "_", filename)[:120]
        path = _BOOK_UPLOAD_DIR / safe
        if not path.exists() or not path.is_relative_to(_BOOK_UPLOAD_DIR):
            raise HTTPException(404, "Book not found")
        path.unlink()
        return {"ok": True}

    @router.get("/adapters")
    async def list_adapters(request: Request):
        """List LoRA adapters produced by training jobs."""
        require_admin(request)
        adapters_dir = Path(DATA_DIR) / "lora_adapters"
        if not adapters_dir.exists():
            return {"adapters": []}
        adapters = []
        for d in sorted(adapters_dir.iterdir()):
            if d.is_dir():
                adapters.append({"name": d.name, "path": str(d)})
        return {"adapters": adapters}

    return router
