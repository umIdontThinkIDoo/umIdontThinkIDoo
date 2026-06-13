# src/ollama_control.py
"""Start/stop/status control for a LOCAL Ollama server.

HARD SECURITY RULE: the managed server binds 127.0.0.1 only. ``OLLAMA_HOST`` is
forced to ``127.0.0.1:11434`` here and is never derived from user input, so the
"Start Ollama" button can never expose the model server off-box (no 0.0.0.0,
no tunnel).

Model-store gotcha (see HANDOFF): the systemd unit runs ``ollama`` as the
``ollama`` user with ``OLLAMA_MODELS`` on a removable drive that the service
user often cannot traverse. We instead launch ``ollama serve`` as the *current*
user and point ``OLLAMA_MODELS`` at the existing store so the already-pulled
models are visible. The store path is resolved (never hardcoded) from, in order:
the ``ollama_models_path`` setting, the ``OLLAMA_MODELS`` env var, or the path
declared in the installed systemd unit.
"""
import os
import re
import time
import shutil
import signal
import logging
import subprocess

import httpx

from core.platform_compat import pid_alive, kill_process_tree, detached_popen_kwargs
from src.constants import DATA_DIR
from src.settings import get_setting

logger = logging.getLogger(__name__)

# Loopback only. Not configurable — this is the security guarantee.
OLLAMA_HOST = "127.0.0.1:11434"
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}"

_PID_FILE = os.path.join(DATA_DIR, "ollama.pid")
_LOG_FILE = os.path.join(DATA_DIR, "logs", "ollama.log")

_SYSTEMD_UNIT_PATHS = (
    "/etc/systemd/system/ollama.service",
    "/usr/lib/systemd/system/ollama.service",
)


def _ollama_binary() -> str | None:
    """Absolute path to the ollama binary, or None if not installed."""
    found = shutil.which("ollama")
    if found:
        return found
    for cand in ("/usr/local/bin/ollama", "/usr/bin/ollama"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _models_path_from_systemd() -> str | None:
    """Parse ``Environment="OLLAMA_MODELS=..."`` out of the installed unit."""
    for path in _SYSTEMD_UNIT_PATHS:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        m = re.search(r'OLLAMA_MODELS=(?:")?([^"\n]+)', text)
        if m:
            return m.group(1).strip()
    return None


def resolve_models_path() -> str | None:
    """Resolve the model store, honoring explicit config first.

    Priority: ``ollama_models_path`` setting → ``OLLAMA_MODELS`` env →
    systemd unit declaration → None (let ollama use its own default).
    """
    configured = (get_setting("ollama_models_path", "") or "").strip()
    if configured:
        return configured
    env = (os.environ.get("OLLAMA_MODELS") or "").strip()
    if env:
        return env
    return _models_path_from_systemd()


def _read_pid() -> int | None:
    try:
        with open(_PID_FILE, "r", encoding="ascii") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    from core.atomic_io import atomic_write_text
    atomic_write_text(_PID_FILE, str(pid))


def _clear_pid() -> None:
    try:
        os.remove(_PID_FILE)
    except OSError:
        pass


def _probe(timeout: float = 1.5) -> dict | None:
    """Return Ollama's /api/version payload if reachable, else None."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/version", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except (httpx.HTTPError, ValueError):
        pass
    return None


def status(timeout: float = 1.5) -> dict:
    """Current Ollama status.

    ``managed`` is True only when the live server is the process we started
    (we hold its PID). A server we didn't start (e.g. the systemd service) reads
    as running-but-not-managed, and stop() will decline to kill it.
    """
    version = _probe(timeout=timeout)
    running = version is not None
    pid = _read_pid()
    managed = bool(running and pid and pid_alive(pid))
    if pid and not pid_alive(pid):
        _clear_pid()
        pid = None
    return {
        "running": running,
        "managed": managed,
        "pid": pid if managed else None,
        "version": (version or {}).get("version"),
        "base_url": OLLAMA_BASE_URL,
        "installed": _ollama_binary() is not None,
        "models_path": resolve_models_path(),
    }


def start(wait_timeout: float = 20.0) -> dict:
    """Launch ``ollama serve`` bound to loopback, then wait until it answers.

    Idempotent: if a server is already reachable, returns its status with
    ``started=False``. Returns ``{"ok": False, "error": ...}`` on failure.
    """
    existing = _probe()
    if existing is not None:
        st = status()
        st["ok"] = True
        st["started"] = False
        st["message"] = "Ollama is already running."
        return st

    binary = _ollama_binary()
    if not binary:
        return {"ok": False, "started": False, "error": "ollama binary not found on PATH."}

    env = dict(os.environ)
    env["OLLAMA_HOST"] = OLLAMA_HOST  # loopback only — never overridden
    models_path = resolve_models_path()
    if models_path:
        env["OLLAMA_MODELS"] = models_path

    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
    try:
        log_fh = open(_LOG_FILE, "ab")
    except OSError:
        log_fh = subprocess.DEVNULL

    try:
        proc = subprocess.Popen(
            [binary, "serve"],
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **detached_popen_kwargs(),
        )
    except OSError as e:
        return {"ok": False, "started": False, "error": f"Failed to launch ollama: {e}"}
    finally:
        if log_fh not in (subprocess.DEVNULL,):
            try:
                log_fh.close()
            except OSError:
                pass

    _write_pid(proc.pid)

    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if not pid_alive(proc.pid):
            _clear_pid()
            return {
                "ok": False,
                "started": False,
                "error": f"ollama serve exited immediately; see {_LOG_FILE}.",
            }
        if _probe(timeout=1.0) is not None:
            st = status()
            st["ok"] = True
            st["started"] = True
            st["message"] = "Ollama started."
            return st
        time.sleep(0.4)

    return {
        "ok": False,
        "started": True,
        "pid": proc.pid,
        "error": f"ollama started (pid {proc.pid}) but did not become ready within {int(wait_timeout)}s.",
    }


def stop() -> dict:
    """Stop the Ollama server we started.

    Refuses to kill a server we did not launch (e.g. the systemd service),
    since that is an externally-managed process the user controls elsewhere.
    """
    pid = _read_pid()
    if pid and pid_alive(pid):
        kill_process_tree(pid)
        # Give it a moment to exit, then confirm.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and pid_alive(pid):
            time.sleep(0.2)
        if pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        _clear_pid()
        return {"ok": True, "stopped": True, "message": "Ollama stopped."}

    _clear_pid()
    if _probe() is not None:
        return {
            "ok": False,
            "stopped": False,
            "error": "Ollama is running but was not started by Odysseus "
                     "(likely the systemd service); refusing to stop it.",
        }
    return {"ok": True, "stopped": False, "message": "Ollama was not running."}
