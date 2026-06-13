"""The model-driven personalizer interview route (/api/memory/interview).

Replaces the old fixed question list: the model is handed the transcript so far
and returns the next adaptive question, or signals completion. Must strip
<think> output, honour the [INTERVIEW_COMPLETE] sentinel, 400 with no model,
and 502 on LLM failure.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import routes.memory_routes as mr


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _router(monkeypatch, *, reply=None, boom=False, endpoint=("http://x", "m", {})):
    monkeypatch.setattr(mr, "get_current_user", lambda request: "alice", raising=False)
    monkeypatch.setattr(mr, "require_user", lambda request: "alice", raising=False)
    monkeypatch.setattr(mr, "resolve_endpoint", lambda which, owner=None: endpoint)

    captured = {}

    async def _fake_llm(url, model, messages, **kwargs):
        captured["messages"] = messages
        if boom:
            raise RuntimeError("model down")
        return reply

    monkeypatch.setattr(mr, "llm_call_async", _fake_llm)
    router = mr.setup_memory_routes(MagicMock(), MagicMock())
    return _route(router, "/api/memory/interview", "POST"), captured


def _req(body):
    async def _json():
        return body
    return SimpleNamespace(json=_json)


def test_returns_next_question(monkeypatch):
    interview, cap = _router(monkeypatch, reply="What do you do for work?")
    out = asyncio.run(interview(request=_req({"transcript": [
        {"role": "assistant", "content": "Hi! What's your name?"},
        {"role": "user", "content": "Dari"},
    ]})))
    assert out == {"question": "What do you do for work?", "done": False}
    # System prompt + the two transcript turns are forwarded to the model.
    roles = [m["role"] for m in cap["messages"]]
    assert roles == ["system", "assistant", "user"]


def test_strips_think_block(monkeypatch):
    interview, _ = _router(monkeypatch, reply="<think>plan</think>What are your goals?")
    out = asyncio.run(interview(request=_req({"transcript": []})))
    assert out["question"] == "What are your goals?"
    assert out["done"] is False


def test_sentinel_marks_done(monkeypatch):
    interview, _ = _router(monkeypatch, reply="[INTERVIEW_COMPLETE]")
    out = asyncio.run(interview(request=_req({"transcript": [{"role": "user", "content": "x"}]})))
    assert out["done"] is True
    assert out["question"] == ""


def test_cold_start_nudges_model(monkeypatch):
    interview, cap = _router(monkeypatch, reply="First question?")
    asyncio.run(interview(request=_req({"transcript": []})))
    # Empty transcript → a kickoff user turn is appended so the model opens.
    assert cap["messages"][-1]["role"] == "user"
    assert len(cap["messages"]) == 2  # system + kickoff


def test_no_model_configured_400(monkeypatch):
    interview, _ = _router(monkeypatch, reply="x", endpoint=(None, None, {}))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(interview(request=_req({"transcript": []})))
    assert exc.value.status_code == 400


def test_llm_failure_502(monkeypatch):
    interview, _ = _router(monkeypatch, boom=True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(interview(request=_req({"transcript": []})))
    assert exc.value.status_code == 502


def test_empty_reply_is_502_not_silent_done(monkeypatch):
    # A model that returns empty content (a hiccup) must NOT be treated as
    # "interview complete" — it should 502 so the frontend falls back.
    interview, _ = _router(monkeypatch, reply="")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(interview(request=_req({"transcript": [{"role": "user", "content": "hi"}]})))
    assert exc.value.status_code == 502


def test_reply_that_is_only_think_block_is_502(monkeypatch):
    # If the whole reply was reasoning that strips to empty, that's also a hiccup.
    interview, _ = _router(monkeypatch, reply="<think>still thinking…</think>")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(interview(request=_req({"transcript": [{"role": "user", "content": "hi"}]})))
    assert exc.value.status_code == 502
