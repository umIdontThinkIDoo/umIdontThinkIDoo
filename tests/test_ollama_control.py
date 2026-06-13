"""Tests for src/ollama_control.py and the /api/ollama/* routes.

These never launch a real ollama; the subprocess + probe layers are mocked so
the logic (loopback enforcement, model-path resolution, idempotency, refusing to
kill an external server) is what's under test.
"""
import asyncio
import json

import pytest

from src import ollama_control as oc
from routes import model_routes


# ───────────────────────── model-path resolution ─────────────────────────

def test_resolve_models_path_prefers_setting(monkeypatch):
    monkeypatch.setattr(oc, "get_setting", lambda k, d=None: "/from/setting")
    monkeypatch.setenv("OLLAMA_MODELS", "/from/env")
    assert oc.resolve_models_path() == "/from/setting"


def test_resolve_models_path_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(oc, "get_setting", lambda k, d=None: "")
    monkeypatch.setenv("OLLAMA_MODELS", "/from/env")
    assert oc.resolve_models_path() == "/from/env"


def test_resolve_models_path_reads_systemd(monkeypatch):
    monkeypatch.setattr(oc, "get_setting", lambda k, d=None: "")
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(oc, "_models_path_from_systemd", lambda: "/from/systemd")
    assert oc.resolve_models_path() == "/from/systemd"


# ───────────────────────────────── status ────────────────────────────────

def test_status_running_and_managed(monkeypatch):
    monkeypatch.setattr(oc, "_probe", lambda timeout=1.5: {"version": "0.24.0"})
    monkeypatch.setattr(oc, "_read_pid", lambda: 4242)
    monkeypatch.setattr(oc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "/usr/local/bin/ollama")
    st = oc.status()
    assert st["running"] is True
    assert st["managed"] is True
    assert st["pid"] == 4242
    assert st["version"] == "0.24.0"
    assert st["base_url"] == "http://127.0.0.1:11434"


def test_status_running_but_external_is_not_managed(monkeypatch):
    # Reachable, but no PID file → not ours.
    monkeypatch.setattr(oc, "_probe", lambda timeout=1.5: {"version": "0.24.0"})
    monkeypatch.setattr(oc, "_read_pid", lambda: None)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "/usr/local/bin/ollama")
    st = oc.status()
    assert st["running"] is True
    assert st["managed"] is False
    assert st["pid"] is None


def test_status_clears_stale_pid(monkeypatch):
    cleared = []
    monkeypatch.setattr(oc, "_probe", lambda timeout=1.5: None)
    monkeypatch.setattr(oc, "_read_pid", lambda: 999)
    monkeypatch.setattr(oc, "pid_alive", lambda pid: False)
    monkeypatch.setattr(oc, "_clear_pid", lambda: cleared.append(True))
    monkeypatch.setattr(oc, "_ollama_binary", lambda: None)
    st = oc.status()
    assert st["running"] is False
    assert st["installed"] is False
    assert cleared == [True]


# ───────────────────────────────── start ─────────────────────────────────

def test_start_idempotent_when_already_running(monkeypatch):
    monkeypatch.setattr(oc, "_probe", lambda timeout=1.5: {"version": "0.24.0"})
    monkeypatch.setattr(oc, "_read_pid", lambda: None)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "/usr/local/bin/ollama")
    launched = []
    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **k: launched.append(a) or pytest.fail("should not launch"))
    res = oc.start()
    assert res["ok"] is True
    assert res["started"] is False
    assert launched == []


def test_start_errors_when_binary_missing(monkeypatch):
    monkeypatch.setattr(oc, "_probe", lambda timeout=1.5: None)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: None)
    res = oc.start()
    assert res["ok"] is False
    assert "not found" in res["error"]


def test_start_forces_loopback_host(monkeypatch):
    """The launched process must get OLLAMA_HOST=127.0.0.1:11434 — never 0.0.0.0."""
    captured = {}

    class _FakeProc:
        pid = 5555

    def _fake_popen(argv, env=None, **kwargs):
        captured["argv"] = argv
        captured["env"] = env
        return _FakeProc()

    # First probe (pre-launch) fails; every probe after launch succeeds
    # (the readiness loop and the final status() call both probe).
    probe_calls = {"n": 0}

    def _probe(timeout=1.5):
        probe_calls["n"] += 1
        return None if probe_calls["n"] == 1 else {"version": "0.24.0"}

    monkeypatch.setattr(oc, "_probe", _probe)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "/usr/local/bin/ollama")
    monkeypatch.setattr(oc, "resolve_models_path", lambda: "/models/here")
    monkeypatch.setattr(oc, "_write_pid", lambda pid: None)
    monkeypatch.setattr(oc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(oc, "_read_pid", lambda: 5555)
    monkeypatch.setattr(oc.subprocess, "Popen", _fake_popen)

    res = oc.start()
    assert res["ok"] is True
    assert res["started"] is True
    assert captured["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert "0.0.0.0" not in captured["env"]["OLLAMA_HOST"]
    assert captured["env"]["OLLAMA_MODELS"] == "/models/here"
    assert captured["argv"][0] == "/usr/local/bin/ollama"
    assert captured["argv"][1] == "serve"


def test_start_reports_immediate_exit(monkeypatch):
    monkeypatch.setattr(oc, "_probe", lambda timeout=1.5: None)
    monkeypatch.setattr(oc, "_ollama_binary", lambda: "/usr/local/bin/ollama")
    monkeypatch.setattr(oc, "resolve_models_path", lambda: None)
    monkeypatch.setattr(oc, "_write_pid", lambda pid: None)
    monkeypatch.setattr(oc, "_clear_pid", lambda: None)
    monkeypatch.setattr(oc, "pid_alive", lambda pid: False)  # died right away

    class _FakeProc:
        pid = 6666

    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **k: _FakeProc())
    res = oc.start()
    assert res["ok"] is False
    assert "exited immediately" in res["error"]


# ───────────────────────────────── stop ──────────────────────────────────

def test_stop_kills_managed_process(monkeypatch):
    killed = []
    monkeypatch.setattr(oc, "_read_pid", lambda: 7777)
    # Alive on the first liveness check (so we issue the kill), dead on every
    # check after that (the wait loop and post-loop check both call pid_alive).
    alive_calls = {"n": 0}

    def _pid_alive(pid):
        alive_calls["n"] += 1
        return alive_calls["n"] == 1

    monkeypatch.setattr(oc, "pid_alive", _pid_alive)
    monkeypatch.setattr(oc, "kill_process_tree", lambda pid: killed.append(pid))
    monkeypatch.setattr(oc, "_clear_pid", lambda: None)
    res = oc.stop()
    assert res["ok"] is True
    assert res["stopped"] is True
    assert killed == [7777]


def test_stop_refuses_external_server(monkeypatch):
    monkeypatch.setattr(oc, "_read_pid", lambda: None)
    monkeypatch.setattr(oc, "_clear_pid", lambda: None)
    monkeypatch.setattr(oc, "_probe", lambda timeout=1.5: {"version": "0.24.0"})
    res = oc.stop()
    assert res["ok"] is False
    assert res["stopped"] is False
    assert "not started by Odysseus" in res["error"]


def test_stop_when_not_running(monkeypatch):
    monkeypatch.setattr(oc, "_read_pid", lambda: None)
    monkeypatch.setattr(oc, "_clear_pid", lambda: None)
    monkeypatch.setattr(oc, "_probe", lambda timeout=1.5: None)
    res = oc.stop()
    assert res["ok"] is True
    assert res["stopped"] is False


# ───────────────────────────────── routes ────────────────────────────────

def _get_route(path, method):
    router = model_routes.setup_model_routes(model_discovery=None)
    for route in router.routes:
        if getattr(route, "path", "") == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"{method} {path} not found")


class _Req:
    pass


def test_route_status(monkeypatch):
    monkeypatch.setattr(model_routes, "require_admin", lambda request: None)
    monkeypatch.setattr(oc, "status", lambda *a, **k: {"running": False, "managed": False})
    endpoint = _get_route("/api/ollama/status", "GET")
    result = asyncio.run(endpoint(_Req()))
    assert result == {"running": False, "managed": False}


def test_route_start_raises_503_on_failure(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setattr(model_routes, "require_admin", lambda request: None)
    monkeypatch.setattr(oc, "start", lambda *a, **k: {"ok": False, "error": "boom"})
    endpoint = _get_route("/api/ollama/start", "POST")
    with pytest.raises(HTTPException) as ei:
        asyncio.run(endpoint(_Req()))
    assert ei.value.status_code == 503
    assert ei.value.detail == "boom"


def test_route_start_ok(monkeypatch):
    monkeypatch.setattr(model_routes, "require_admin", lambda request: None)
    monkeypatch.setattr(oc, "start", lambda *a, **k: {"ok": True, "started": True})
    endpoint = _get_route("/api/ollama/start", "POST")
    result = asyncio.run(endpoint(_Req()))
    assert result["started"] is True


def test_route_stop_raises_409_on_failure(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setattr(model_routes, "require_admin", lambda request: None)
    monkeypatch.setattr(oc, "stop", lambda *a, **k: {"ok": False, "error": "external"})
    endpoint = _get_route("/api/ollama/stop", "POST")
    with pytest.raises(HTTPException) as ei:
        asyncio.run(endpoint(_Req()))
    assert ei.value.status_code == 409
    assert ei.value.detail == "external"
