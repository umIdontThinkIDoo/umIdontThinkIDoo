"""Process-wide "heavy ingest in progress" gate.

ONE shared signal that lets the chat path know a GPU-saturating ingest or
knowledge-graph backfill is running, so chat can pause until it finishes (or the
user interrupts). The user's intent: while the library is being embedded /
enriched the GPUs run flat-out, and chat stays out of the way rather than fight
for VRAM and crawl.

Design notes:
  * Thread-safe, in-memory. A heavy job lives inside one uvicorn process; a
    restart clears the gate, which is the safe default — nothing stays "locked"
    after a crash.
  * Carries the cooperative STOP flag ("Interrupt ASAP"). A worker checks
    ``stop_requested()`` at safe boundaries (between books / between batches),
    finishes the unit it's on so nothing is abandoned mid-embed, then calls
    ``deactivate()`` and frees the GPU.
  * Best-effort and dependency-light so both the route layer and the chat path
    can import it without pulling in heavy modules.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "active": False,
    "kind": None,          # "upload" | "kg_backfill"
    "label": "",
    "total": 0,
    "done": 0,
    "current": "",
    "started_at": None,
    "stop_requested": False,
}


def activate(kind: str, label: str, total: int = 0) -> None:
    """Open the gate for a new heavy job. Resets progress + stop flag."""
    with _lock:
        _state.update({
            "active": True,
            "kind": kind,
            "label": label,
            "total": int(total or 0),
            "done": 0,
            "current": "",
            "started_at": time.time(),
            "stop_requested": False,
        })


def update(*, done: Optional[int] = None, total: Optional[int] = None,
           current: Optional[str] = None) -> None:
    """Advance progress while a job runs. No-op once the gate is closed."""
    with _lock:
        if not _state["active"]:
            return
        if done is not None:
            _state["done"] = int(done)
        if total is not None:
            _state["total"] = int(total)
        if current is not None:
            _state["current"] = str(current)[:200]


def deactivate() -> None:
    """Close the gate and clear all job state (chat becomes usable again)."""
    with _lock:
        _state.update({
            "active": False,
            "kind": None,
            "label": "",
            "total": 0,
            "done": 0,
            "current": "",
            "started_at": None,
            "stop_requested": False,
        })


def request_stop() -> bool:
    """Ask the running job to halt at its next safe boundary.

    Returns True if a job was active to receive the request, False otherwise.
    """
    with _lock:
        if not _state["active"]:
            return False
        _state["stop_requested"] = True
        return True


def stop_requested() -> bool:
    with _lock:
        return bool(_state["stop_requested"])


def is_active() -> bool:
    with _lock:
        return bool(_state["active"])


def snapshot() -> Dict[str, Any]:
    """A copy of the gate state plus derived ``pct`` / ``elapsed_s`` for the UI."""
    with _lock:
        s = dict(_state)
    s["elapsed_s"] = int(time.time() - s["started_at"]) if (s["active"] and s["started_at"]) else 0
    s["pct"] = min(100, int(100 * s["done"] / s["total"])) if s["total"] > 0 else 0
    return s
