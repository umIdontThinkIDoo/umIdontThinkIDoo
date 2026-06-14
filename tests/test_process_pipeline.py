"""Tests for the process-discipline subsystem (src/process_router.py +
src/process_pipeline.py). Covers the routing heuristic, the off-by-default
contract, the level mechanics, and graceful degradation on stage failure."""
import pytest

from src import process_router as router
from src import process_pipeline as pipeline


# ---- router ----------------------------------------------------------------
def test_trivial_turns_are_none_even_when_forced():
    for msg in ("hi", "hey", "thanks!", "ok", "👍", "got it"):
        assert router.classify_turn(msg) == "none"
        # Forcing full must never escalate a greeting — "hi" can't cost 5x.
        assert router.resolve_level(msg, "full") == "none"
        assert router.resolve_level(msg, "auto") == "none"


def test_engineering_turns_classify_full():
    for msg in (
        "implement a retry decorator in app.py",
        "why does this deadlock?",
        "refactor the auth module",
        "```\ndef f(): pass\n```",
    ):
        assert router.classify_turn(msg) == "full"


def test_explanatory_turns_classify_light():
    assert router.classify_turn("explain how async generators work") == "light"
    assert router.classify_turn("what is the difference between a and b") == "light"


def test_off_setting_is_always_none():
    assert router.resolve_level("implement a parser in main.py", "off") == "none"
    assert router.resolve_level("implement a parser in main.py", None) == "none"


def test_auto_passes_through_classifier():
    assert router.resolve_level("implement a parser in main.py", "auto") == "full"
    assert router.resolve_level("explain recursion", "auto") == "light"


def test_light_setting_floors_substantive_turns():
    # An otherwise-'none' substantive long turn gets floored to light.
    long_plain = "I have been thinking about my weekend plans. " * 20
    assert router.classify_turn(long_plain) in ("none", "light")
    assert router.resolve_level(long_plain, "light") in ("light", "full")
    # A 'full' turn stays full even under the 'light' admin floor.
    assert router.resolve_level("debug this race condition", "light") == "full"


# ---- pipeline: off / none --------------------------------------------------
@pytest.mark.asyncio
async def test_prepare_none_returns_original_messages():
    msgs = [{"role": "user", "content": "hello"}]
    prep = await pipeline.prepare("http://x", "m", msgs, level="none")
    assert prep.level == "none"
    assert prep.messages is msgs
    assert prep.trace == []


@pytest.mark.asyncio
async def test_run_none_is_single_plain_generation(monkeypatch):
    calls = []

    async def fake(url, model, messages, **kwargs):
        calls.append(messages)
        return "plain answer"

    monkeypatch.setattr("src.llm_core.llm_call_async", fake)
    out = await pipeline.run("http://x", "m", [{"role": "user", "content": "hi"}],
                             level="none")
    assert out["answer"] == "plain answer"
    assert out["trace"] == []
    assert len(calls) == 1  # exactly one generation, no extra stages


# ---- pipeline: light -------------------------------------------------------
@pytest.mark.asyncio
async def test_light_extracts_requirements_and_validates(monkeypatch):
    seen_systems = []

    async def fake(url, model, messages, **kwargs):
        sys_txt = " ".join(m["content"] for m in messages if m.get("role") == "system")
        seen_systems.append(sys_txt)
        # Distinguish stages by their system prompt.
        if "requirements-analysis" in sys_txt:
            return "REQUIREMENTS:\n- do the thing"
        if "validation stage" in sys_txt:
            return "✓ did the thing\nConfidence: High — clear"
        return "the answer"

    monkeypatch.setattr("src.llm_core.llm_call_async", fake)
    out = await pipeline.run("http://x", "m",
                             [{"role": "user", "content": "explain X"}], level="light")
    assert out["answer"] == "the answer"
    stages = [t["stage"] for t in out["trace"]]
    assert "requirements" in stages
    assert "validation" in stages
    # The final answer generation must have been guided by the requirements.
    assert any("Process discipline" in s for s in seen_systems)


# ---- pipeline: full --------------------------------------------------------
@pytest.mark.asyncio
async def test_full_runs_draft_critique_revise_validate(monkeypatch):
    stage_order = []

    async def fake(url, model, messages, **kwargs):
        sys_txt = " ".join(m["content"] for m in messages if m.get("role") == "system")
        if "requirements-analysis" in sys_txt:
            stage_order.append("requirements")
            return "REQUIREMENTS:\n- build it"
        if "self-critique stage" in sys_txt:
            stage_order.append("critique")
            return "- missing error handling"
        if "validation stage" in sys_txt:
            stage_order.append("validate")
            return "✓ ok\nConfidence: Medium — fine"
        # Two non-analysis calls: the draft, then the revise (final).
        stage_order.append("gen")
        return "draft-or-final"

    monkeypatch.setattr("src.llm_core.llm_call_async", fake)
    out = await pipeline.run("http://x", "m",
                             [{"role": "user", "content": "implement a thing in x.py"}],
                             level="full")
    assert out["level"] == "full"
    # Requirements precede draft; critique precedes the final; validation last.
    assert stage_order[0] == "requirements"
    assert "critique" in stage_order
    assert stage_order[-1] == "validate"
    assert stage_order.count("gen") == 2  # draft + revise


# ---- pipeline: never breaks a turn -----------------------------------------
@pytest.mark.asyncio
async def test_stage_failure_degrades_gracefully(monkeypatch):
    """If the requirements stage throws, run() still returns a plain answer."""
    state = {"n": 0}

    async def flaky(url, model, messages, **kwargs):
        sys_txt = " ".join(m["content"] for m in messages if m.get("role") == "system")
        if "requirements-analysis" in sys_txt:
            raise RuntimeError("model down")
        return "still answered"

    monkeypatch.setattr("src.llm_core.llm_call_async", flaky)
    out = await pipeline.run("http://x", "m",
                             [{"role": "user", "content": "explain Y"}], level="light")
    # Requirements failed → no trace, but the user still gets an answer.
    assert out["answer"] == "still answered"


@pytest.mark.asyncio
async def test_validate_empty_answer_returns_none(monkeypatch):
    async def fake(url, model, messages, **kwargs):
        return "should not be called"

    monkeypatch.setattr("src.llm_core.llm_call_async", fake)
    ev = await pipeline.validate("http://x", "m", requirements="r",
                                 user_text="u", final_answer="")
    assert ev is None
