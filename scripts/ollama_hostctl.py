#!/usr/bin/env python3
"""Host-side Ollama lifecycle helper for the Dockerized Odysseus deploy.

WHY THIS EXISTS
---------------
Odysseus runs inside a container. A containerized app cannot fork or even see a
process living in the host's network/PID namespace, so the in-app "Start/Stop
Ollama" buttons (src/ollama_control.py) are inert in Docker: there is no ollama
binary in the image and the host's loopback ollama is invisible to the
container. This tiny daemon runs ON THE HOST, owns the ollama lifecycle, and the
container delegates start/stop/status to it over the docker bridge.

SECURITY
--------
- The helper binds the docker-gateway IP only (default 172.17.0.1), never
  0.0.0.0 and never a tunnel. That address is reachable from the host and its
  containers but is NOT routable from the LAN, so the control plane is not
  exposed off-box. This host-internal binding is the deliberate, user-approved
  exception to the "loopback only" rule (the LAN-exposure intent still holds).
- It launches `ollama serve` bound to the same gateway IP (default
  172.17.0.1:11434) so the container can reach the model server — again
  host-internal, not the LAN.
- Every request must carry the shared secret in the `X-Hostctl-Token` header.
  The token lives in a 0600 file shared with the container via the data bind
  mount; it is never committed.

Stdlib only — this runs in the host's plain Python, outside the app venv.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---- Configuration (all overridable via env) --------------------------------
BIND_HOST = os.environ.get("OLLAMA_HOSTCTL_BIND", "172.17.0.1")
BIND_PORT = int(os.environ.get("OLLAMA_HOSTCTL_PORT", "11436"))

# Where ollama itself listens. Host-internal gateway so the container can reach
# it; never 0.0.0.0. Host CLI users can also use this address.
OLLAMA_HOST = os.environ.get("OLLAMA_SERVE_HOST", "172.17.0.1:11434")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}"

_HOME = os.path.expanduser("~")
DATA_DIR = os.environ.get(
    "ODYSSEUS_DATA_DIR", os.path.join(_HOME, "odysseus", "data")
)
TOKEN_FILE = os.environ.get(
    "OLLAMA_HOSTCTL_TOKEN_FILE", os.path.join(DATA_DIR, "ollama_hostctl.token")
)
PID_FILE = os.path.join(DATA_DIR, "ollama.host.pid")
LOG_FILE = os.path.join(DATA_DIR, "logs", "ollama.log")

_SYSTEMD_UNIT_PATHS = (
    "/etc/systemd/system/ollama.service",
    "/usr/lib/systemd/system/ollama.service",
)


# ---- Token ------------------------------------------------------------------
def _load_token() -> str:
    try:
        with open(TOKEN_FILE, "r", encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return ""


# ---- ollama discovery / model store -----------------------------------------
def _ollama_binary() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    for cand in ("/usr/local/bin/ollama", "/usr/bin/ollama"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _models_path_from_systemd() -> str | None:
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


def _resolve_models_path() -> str | None:
    env = (os.environ.get("OLLAMA_MODELS") or "").strip()
    if env:
        return env
    return _models_path_from_systemd()


# ---- pid helpers ------------------------------------------------------------
def _read_pid() -> int | None:
    try:
        with open(PID_FILE, "r", encoding="ascii") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    tmp = f"{PID_FILE}.tmp"
    with open(tmp, "w", encoding="ascii") as f:
        f.write(str(pid))
    os.replace(tmp, PID_FILE)


def _clear_pid() -> None:
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Treat zombies as dead.
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="ascii") as f:
            state = f.read().split(") ", 1)[1].split(" ", 1)[0]
        return state != "Z"
    except OSError:
        return True


# ---- probe ------------------------------------------------------------------
def _probe(timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL}/api/version", timeout=timeout
        ) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError):
        pass
    return None


# ---- lifecycle --------------------------------------------------------------
def do_status(timeout: float = 1.5) -> dict:
    version = _probe(timeout=timeout)
    running = version is not None
    pid = _read_pid()
    managed = bool(running and pid and _pid_alive(pid))
    if pid and not _pid_alive(pid):
        _clear_pid()
        pid = None
    return {
        "running": running,
        "managed": managed,
        "pid": pid if managed else None,
        "version": (version or {}).get("version"),
        "base_url": OLLAMA_BASE_URL,
        "installed": _ollama_binary() is not None,
        "models_path": _resolve_models_path(),
    }


def do_start(wait_timeout: float = 20.0) -> dict:
    if _probe() is not None:
        st = do_status()
        st["ok"] = True
        st["started"] = False
        st["message"] = "Ollama is already running."
        return st

    binary = _ollama_binary()
    if not binary:
        return {"ok": False, "started": False, "error": "ollama binary not found on PATH."}

    env = dict(os.environ)
    env["OLLAMA_HOST"] = OLLAMA_HOST  # host-internal gateway — never the LAN
    models_path = _resolve_models_path()
    if models_path:
        env["OLLAMA_MODELS"] = models_path

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    try:
        log_fh = open(LOG_FILE, "ab")
    except OSError:
        log_fh = subprocess.DEVNULL

    try:
        proc = subprocess.Popen(
            [binary, "serve"],
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from the helper's session
        )
    except OSError as e:
        return {"ok": False, "started": False, "error": f"Failed to launch ollama: {e}"}
    finally:
        if log_fh is not subprocess.DEVNULL:
            try:
                log_fh.close()
            except OSError:
                pass

    _write_pid(proc.pid)

    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if not _pid_alive(proc.pid):
            _clear_pid()
            return {"ok": False, "started": False,
                    "error": f"ollama serve exited immediately; see {LOG_FILE}."}
        if _probe(timeout=1.0) is not None:
            st = do_status()
            st["ok"] = True
            st["started"] = True
            st["message"] = "Ollama started."
            return st
        time.sleep(0.4)

    return {"ok": False, "started": True, "pid": proc.pid,
            "error": f"ollama started (pid {proc.pid}) but was not ready within {int(wait_timeout)}s."}


def do_stop() -> dict:
    pid = _read_pid()
    if pid and _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.2)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        _clear_pid()
        return {"ok": True, "stopped": True, "message": "Ollama stopped."}

    _clear_pid()
    if _probe() is not None:
        return {"ok": False, "stopped": False,
                "error": "Ollama is running but was not started by this helper "
                         "(e.g. the systemd service); refusing to stop it."}
    return {"ok": True, "stopped": False, "message": "Ollama was not running."}


# ---- HTTP -------------------------------------------------------------------
ROUTES = {
    ("GET", "/status"): do_status,
    ("POST", "/start"): do_start,
    ("POST", "/stop"): do_stop,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "ollama-hostctl/1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        token = _load_token()
        if not token:
            return False
        return self.headers.get("X-Hostctl-Token", "") == token

    def _dispatch(self, method: str) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        handler = ROUTES.get((method, path))
        if handler is None:
            self._send(404, {"ok": False, "error": "not found"})
            return
        if not self._authed():
            self._send(403, {"ok": False, "error": "forbidden"})
            return
        try:
            self._send(200, handler())
        except Exception as exc:  # never crash the daemon on a bad request
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        # Drain any request body so the socket stays clean.
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
        except (ValueError, OSError):
            pass
        self._dispatch("POST")

    def log_message(self, *args):  # quieter logs
        sys.stderr.write("[ollama-hostctl] " + (args[0] % args[1:]) + "\n")


def main() -> int:
    if not _load_token():
        sys.stderr.write(
            f"[ollama-hostctl] no token at {TOKEN_FILE}; refusing to start "
            "(generate one: python3 -c 'import secrets;print(secrets.token_urlsafe(32))' > "
            f"{TOKEN_FILE}; chmod 600 {TOKEN_FILE})\n"
        )
        return 2
    httpd = ThreadingHTTPServer((BIND_HOST, BIND_PORT), Handler)
    sys.stderr.write(
        f"[ollama-hostctl] listening on http://{BIND_HOST}:{BIND_PORT} "
        f"(ollama -> {OLLAMA_BASE_URL})\n"
    )

    # Optionally bring ollama up as soon as the helper starts, so a host reboot
    # leaves the model server running without anyone clicking "Start". Done in a
    # background thread so the control HTTP server is responsive immediately.
    if os.environ.get("OLLAMA_HOSTCTL_AUTOSTART", "").strip().lower() in ("1", "true", "yes", "on"):
        import threading

        def _boot_ollama() -> None:
            res = do_start()
            sys.stderr.write(
                "[ollama-hostctl] autostart: "
                + str(res.get("message") or res.get("error") or res)
                + "\n"
            )

        threading.Thread(target=_boot_ollama, name="autostart", daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
