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


# ------------------------------ per-GPU job lock ----------------------------

@pytest.fixture
def launcher(client, monkeypatch):
    """Client plus a fake Popen that records each launch's env. pid_alive is
    forced True so started jobs stay 'running' for the conflict checks."""
    launches = []

    class _FakeProc:
        pid = 5000

    def _fake_popen(cmd, *a, **k):
        launches.append({"cmd": cmd, "env": k.get("env")})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    monkeypatch.setattr(tr, "pid_alive", lambda pid: True)

    def start(gpu=None):
        return client.post(
            "/api/training/start",
            json={"job_type": "sft", "model_id": "m", "gpu": gpu},
            headers=_ADMIN,
        )

    return start, launches


def test_pinned_start_sets_cuda_visible_devices(launcher):
    start, launches = launcher
    assert start(gpu=1).status_code == 200
    assert launches[0]["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_unpinned_start_inherits_parent_gpu_visibility(launcher):
    """gpu=None must not invent a CUDA_VISIBLE_DEVICES value — the job spans
    whatever the parent process can see."""
    import os
    start, launches = launcher
    assert start(gpu=None).status_code == 200
    assert launches[0]["env"].get("CUDA_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES")


def test_same_gpu_start_conflicts_409(launcher):
    start, _ = launcher
    assert start(gpu=0).status_code == 200
    r = start(gpu=0)
    assert r.status_code == 409
    assert "GPU 0" in r.json()["detail"]


def test_different_gpus_run_concurrently(launcher, client):
    start, launches = launcher
    assert start(gpu=0).status_code == 200
    assert start(gpu=1).status_code == 200
    assert len(launches) == 2
    state = client.get("/api/training/state", headers=_ADMIN).json()
    assert [j["status"] for j in state["jobs"]] == ["running", "running"]


def test_all_gpu_job_blocks_pinned_start(launcher):
    start, _ = launcher
    assert start(gpu=None).status_code == 200
    r = start(gpu=0)
    assert r.status_code == 409
    assert "all GPUs" in r.json()["detail"]


def test_pinned_job_blocks_all_gpu_start(launcher):
    start, _ = launcher
    assert start(gpu=1).status_code == 200
    assert start(gpu=None).status_code == 409


# ---------------------------- stale-job reaping -----------------------------

def _flip_pid_alive_dead(monkeypatch):
    monkeypatch.setattr(tr, "pid_alive", lambda pid: False)


def test_dead_job_without_marker_is_reaped_to_error_and_unblocks_start(
        launcher, client, monkeypatch, tmp_path):
    """A crashed job (dead pid, no [TRAINING_OK] in its log) must not hold the
    GPU lock forever: /start itself refreshes liveness, flips the job to
    'error', and lets the new job through."""
    start, _ = launcher
    old_id = start(gpu=0).json()["job_id"]
    (tmp_path / "training_logs" / f"{old_id}.log").write_text("Traceback ...")

    _flip_pid_alive_dead(monkeypatch)
    r = start(gpu=0)
    assert r.status_code == 200, r.text

    jobs = {j["id"]: j for j in tr._load_state()["jobs"]}
    assert jobs[old_id]["status"] == "error"
    assert jobs[r.json()["job_id"]]["status"] == "running"


def test_dead_job_with_ok_marker_is_reaped_to_done(launcher, client, monkeypatch, tmp_path):
    start, _ = launcher
    old_id = start(gpu=0).json()["job_id"]
    (tmp_path / "training_logs" / f"{old_id}.log").write_text("...\n[TRAINING_OK]\n")

    _flip_pid_alive_dead(monkeypatch)
    state = client.get("/api/training/state", headers=_ADMIN).json()
    assert {j["id"]: j["status"] for j in state["jobs"]}[old_id] == "done"


def test_dead_job_without_log_is_reaped_to_error(launcher, client, monkeypatch):
    start, _ = launcher
    old_id = start(gpu=0).json()["job_id"]

    _flip_pid_alive_dead(monkeypatch)
    state = client.get("/api/training/state", headers=_ADMIN).json()
    assert {j["id"]: j["status"] for j in state["jobs"]}[old_id] == "error"


# -------------------------------- GET /gpus ---------------------------------

def test_gpus_endpoint_parses_nvidia_smi_csv(client, monkeypatch):
    class _Out:
        returncode = 0
        stdout = (
            "0, NVIDIA GeForce RTX 4060, 8188, 8.9\n"
            "1, Tesla P100-PCIE-16GB, 16276, 6.0\n"
        )

    monkeypatch.setattr(tr.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Out())

    gpus = client.get("/api/training/gpus", headers=_ADMIN).json()["gpus"]
    assert [g["index"] for g in gpus] == [0, 1]
    # bf16 needs Ampere+ (cc 8.0+); Pascal (6.0) must read as fp16-only.
    assert gpus[0]["bf16"] is True
    assert gpus[1]["bf16"] is False
    assert gpus[1]["memory_mb"] == 16276


def test_gpus_endpoint_degrades_to_empty_without_nvidia_smi(client, monkeypatch):
    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    assert client.get("/api/training/gpus", headers=_ADMIN).json() == {"gpus": []}
