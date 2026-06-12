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
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.middleware import require_admin
from core.platform_compat import kill_process_tree, pid_alive, detached_popen_kwargs
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
    # Which physical GPU to pin this job to (0, 1, ...). None = let the job span
    # all visible GPUs (device_map="auto"). On a heterogeneous box (e.g. a fast
    # Ada card + a slow Pascal card) you want per-GPU jobs, so the launcher sets
    # CUDA_VISIBLE_DEVICES=<gpu> and only this one card is visible to the run.
    gpu: Optional[int] = None


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

    # Use the interpreter Odysseus is running under (the venv Python) rather
    # than a bare "python" off PATH: many hosts only ship python3, and even
    # where "python" resolves it may be the system interpreter without the
    # training deps installed. sys.executable bypasses PATH entirely — same
    # rationale as Cookbook's local serve/scan path.
    cmd = [
        sys.executable, str(script),
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
    # qlora is a merge step (base model + adapter), not a training run, so its
    # script intentionally has no --dataset flag. Passing it would make argparse
    # reject the whole invocation and the job would die immediately.
    if req.dataset_path and req.job_type != "qlora":
        cmd += ["--dataset", req.dataset_path]
    if req.book_filename and req.job_type == "rl_loop":
        book_path = _BOOK_UPLOAD_DIR / req.book_filename
        cmd += ["--book", str(book_path)]
    if req.base_adapter_path and req.job_type == "qlora":
        cmd += ["--base-adapter", req.base_adapter_path]
    if req.output_name:
        cmd += ["--output-name", req.output_name]
    return cmd


def _gpu_env(gpu: Optional[int]) -> dict:
    """Subprocess env that pins the job to one physical GPU.

    Inherits the parent env (so the venv interpreter still finds its deps) and
    overlays CUDA_VISIBLE_DEVICES=<gpu>. With a single device visible, the
    scripts' device_map="auto" / get_device_name(0) all target that one card —
    which is exactly the per-GPU-job model on a heterogeneous box. gpu=None
    leaves CUDA_VISIBLE_DEVICES untouched (job may span all GPUs).
    """
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _list_gpus() -> list[dict]:
    """Enumerate NVIDIA GPUs via nvidia-smi (no torch dependency).

    Returns [] if nvidia-smi is absent or errors, so the UI degrades to "no GPU
    selector" rather than 500ing. compute_cap lets the UI warn about Pascal
    (<7.0: no bf16, Unsloth unsupported — but QLoRA/fp16 still works).
    """
    import subprocess
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, name, mem, cc = parts[0], parts[1], parts[2], parts[3]
        try:
            gpus.append({
                "index": int(idx),
                "name": name,
                "memory_mb": int(float(mem)),
                "compute_cap": cc,
                "bf16": float(cc) >= 8.0,   # bf16 needs Ampere+ (cc 8.0+)
            })
        except ValueError:
            continue
    return gpus


def setup_training_routes() -> APIRouter:
    router = APIRouter(prefix="/api/training", tags=["training"])

    @router.get("/gpus")
    async def list_gpus(request: Request):
        require_admin(request)
        return {"gpus": _list_gpus()}

    def _refresh_job_liveness(state: dict) -> None:
        """Mark 'running' jobs whose pid is gone as done/error (by log marker).

        Every reader of the running-jobs list must do this first — otherwise a
        job that crashed (or died with the machine) stays 'running' in the
        state file forever and blocks new starts with a phantom 409.
        """
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

    @router.get("/state")
    async def get_training_state(request: Request):
        require_admin(request)
        state = _load_state()
        _refresh_job_liveness(state)
        _save_state(state)
        return state

    @router.post("/start")
    async def start_training(request: Request, body: TrainingRequest):
        require_admin(request)
        state = _load_state()
        _refresh_job_liveness(state)

        # Prevent jobs that would contend for the same GPU. Per-GPU independent
        # jobs are allowed (e.g. a QLoRA run on GPU 1 while a small SFT runs on
        # GPU 0). A job that pins no GPU (gpu=None) spans all cards, so it
        # conflicts with anything; a pinned job conflicts only with a job on the
        # same card or with an all-GPU job.
        for job in state.get("jobs", []):
            if job.get("status") != "running":
                continue
            running_gpu = job.get("gpu")
            if body.gpu is None or running_gpu is None or running_gpu == body.gpu:
                where = "all GPUs" if running_gpu is None else f"GPU {running_gpu}"
                raise HTTPException(
                    409,
                    f"A training job is already running on {where}. "
                    "Stop it first, or pin this job to a free GPU.",
                )

        # Fail fast with a clear message instead of launching a subprocess that
        # would immediately exit because a required script arg is missing.
        if body.job_type == "qlora" and not body.base_adapter_path:
            raise HTTPException(400, "QLoRA merge requires base_adapter_path (the adapter to merge).")
        if body.job_type == "rl_loop" and not body.book_filename:
            raise HTTPException(400, "RL Loop requires book_filename (upload a book first).")

        job_id = str(uuid.uuid4())[:8]
        log_path = str(_job_log_path(job_id))

        try:
            cmd = _build_train_script(body, job_id, log_path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        import subprocess
        kwargs = detached_popen_kwargs()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
                env=_gpu_env(body.gpu),
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
            "gpu": body.gpu,
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
