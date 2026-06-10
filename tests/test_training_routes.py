"""Regression tests for routes/training_routes.py.

These cover the train-script construction and the /start validation/launch
path. The training scripts themselves need heavy ML deps (torch/trl/peft) that
the app does not install, so the subprocess launch is mocked — we assert on the
exact command Odysseus would run, not on a real training run.
"""
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
import routes.training_routes as tr
from routes.training_routes import TrainingRequest, _build_train_script, setup_training_routes


_ADMIN = {INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect all on-disk state into a temp dir so tests don't touch ./data.
    monkeypatch.setattr(tr, "_TRAINING_STATE_PATH", tmp_path / "training_state.json")
    monkeypatch.setattr(tr, "_TRAINING_LOG_DIR", tmp_path / "training_logs")
    monkeypatch.setattr(tr, "_BOOK_UPLOAD_DIR", tmp_path / "training_books")
    app = FastAPI()
    app.include_router(setup_training_routes())
    return TestClient(app)


# --------------------------- _build_train_script ---------------------------

def test_build_uses_current_interpreter_not_bare_python():
    """The launcher must use sys.executable, never a bare 'python' off PATH:
    many hosts only ship python3, and even where 'python' resolves it may lack
    the venv's training deps."""
    req = TrainingRequest(job_type="sft", model_id="m")
    cmd = _build_train_script(req, "job1", "/tmp/job1.log")
    assert cmd[0] == sys.executable
    assert "python" not in cmd[0].rsplit("/", 1)[-1:] or cmd[0] == sys.executable


def test_sft_includes_dataset_flag():
    req = TrainingRequest(job_type="sft", model_id="m", dataset_path="/data/ds")
    cmd = _build_train_script(req, "j", "/tmp/j.log")
    assert "--dataset" in cmd
    assert cmd[cmd.index("--dataset") + 1] == "/data/ds"


def test_qlora_never_gets_dataset_flag():
    """qlora_merge.py has no --dataset arg; passing it makes argparse reject the
    whole invocation and the job dies instantly."""
    req = TrainingRequest(
        job_type="qlora", model_id="m",
        dataset_path="/data/ds", base_adapter_path="/data/adapter",
    )
    cmd = _build_train_script(req, "j", "/tmp/j.log")
    assert "--dataset" not in cmd
    assert "--base-adapter" in cmd


def test_rl_loop_passes_book():
    req = TrainingRequest(job_type="rl_loop", model_id="m", book_filename="book.pdf")
    cmd = _build_train_script(req, "j", "/tmp/j.log")
    assert "--book" in cmd


def test_unknown_job_type_raises():
    with pytest.raises(ValueError):
        _build_train_script(TrainingRequest(job_type="bogus", model_id="m"), "j", "/tmp/j.log")


# ------------------------------- /start route ------------------------------

def test_qlora_without_adapter_is_rejected(client):
    r = client.post("/api/training/start",
                    json={"job_type": "qlora", "model_id": "m"}, headers=_ADMIN)
    assert r.status_code == 400
    assert "base_adapter_path" in r.json()["detail"]


def test_rl_loop_without_book_is_rejected(client):
    r = client.post("/api/training/start",
                    json={"job_type": "rl_loop", "model_id": "m"}, headers=_ADMIN)
    assert r.status_code == 400
    assert "book" in r.json()["detail"].lower()


def test_start_requires_admin(client):
    # No internal-tool header and auth defaults on -> 403.
    r = client.post("/api/training/start", json={"job_type": "sft", "model_id": "m"})
    assert r.status_code == 403


def test_start_launches_with_venv_interpreter(client, monkeypatch):
    """End-to-end: a valid /start mocks the subprocess and asserts Odysseus
    launches the script with sys.executable and records a running job."""
    captured = {}

    class _FakeProc:
        pid = 4242

    def _fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    # Keep the (fake) pid "alive" so /state's liveness refresh doesn't downgrade
    # the job — we're testing launch wiring, not process supervision.
    monkeypatch.setattr(tr, "pid_alive", lambda pid: True)
    r = client.post("/api/training/start",
                    json={"job_type": "sft", "model_id": "demo"}, headers=_ADMIN)
    assert r.status_code == 200, r.text
    assert captured["cmd"][0] == sys.executable

    state = client.get("/api/training/state", headers=_ADMIN).json()
    assert state["jobs"][0]["status"] == "running"
    assert state["jobs"][0]["model_id"] == "demo"
