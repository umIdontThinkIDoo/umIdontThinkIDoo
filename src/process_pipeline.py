"""Process-discipline pipeline — Claude-like Reason→Draft→Critique→Revise→Validate
around a single, ordinary model.

WHY
---
A smaller local model (Qwen3-30B-A3B, hermes3, gpt-oss, …) gets a large share of
Claude's perceived "smartness" not from parameters but from *process discipline*:
extracting the real requirements before answering, drafting, critiquing its own
draft, revising, and validating that it actually did the thing. This module
orchestrates that discipline as extra generations around the model the user
already selected.

UNIVERSAL & SAME-MODEL
----------------------
Every stage calls the SAME endpoint/model the turn already uses (``llm_call_async``),
so there is no second model to load and nothing model-specific. It works for any
provider.

LEVELS (chosen per-turn by src/process_router.py)
    "none"  — pipeline does nothing; caller generates normally.
    "light" — extract requirements (guide the single answer) + validate after.
    "full"  — requirements + draft + self-critique, then the FINAL answer is a
              revise pass, followed by validation.

STREAMING-FRIENDLY
------------------
``prepare()`` runs the non-streamed pre-stages and returns the augmented message
list plus a trace, so the caller can stream the FINAL answer normally (the only
part the user watches token-by-token). ``validate()`` runs after the stream to
append the completion checklist. ``run()`` is the all-in-one non-streaming
variant for the plain ``/api/chat`` path.

DEFENSIVE
---------
Any stage failure degrades gracefully to the plain answer — the pipeline must
never break a chat turn. Every public coroutine swallows its own errors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Per-stage generation caps. Analysis stages are short; the final answer is not
# capped here (the caller's own max_tokens governs it).
_REQUIREMENTS_MAX_TOKENS = 500
_CRITIQUE_MAX_TOKENS = 700
_DRAFT_MAX_TOKENS = 1500
_VALIDATE_MAX_TOKENS = 500

# Lower temperature for the analytical stages — we want disciplined, not creative.
_ANALYSIS_TEMPERATURE = 0.3


@dataclass
class TraceEvent:
    """One collapsible process-trace entry surfaced to the client."""
    stage: str          # "requirements" | "critique" | "validation"
    title: str
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"type": "process_trace", "stage": self.stage,
                "title": self.title, "content": self.content}


@dataclass
class PipelinePrep:
    """Result of the pre-stream stages."""
    messages: List[Dict]                       # augmented messages for the FINAL gen
    trace: List[TraceEvent] = field(default_factory=list)
    requirements: str = ""                     # carried into validate()
    level: str = "none"


# ---- prompts (model-agnostic) ----------------------------------------------
_REQUIREMENTS_SYSTEM = (
    "You are the requirements-analysis stage of a careful engineering assistant. "
    "Given the latest user request (with prior context), extract what is ACTUALLY "
    "being asked before any answer is written. Be concise and concrete. Output "
    "exactly these three sections and nothing else:\n\n"
    "REQUIREMENTS:\n- (each explicit thing the answer must do)\n"
    "ASSUMPTIONS:\n- (each thing you are assuming because it was unstated)\n"
    "SUCCESS CRITERIA:\n- (how to tell the final answer is correct/complete)\n\n"
    "If the request is simple, keep it to a couple of bullets each. Do not answer "
    "the request itself."
)

_CRITIQUE_SYSTEM = (
    "You are the self-critique stage of a careful engineering assistant, reviewing "
    "a DRAFT answer as a skeptical principal engineer would. Do not rewrite it. "
    "Identify, specifically and briefly:\n"
    "- Requirements or success criteria the draft missed or only partly met\n"
    "- Architectural weaknesses, wrong assumptions, or simpler approaches\n"
    "- Missing error handling, edge cases, tests, or security/production concerns\n"
    "- Anything factually shaky or likely to be misunderstood\n"
    "If the draft is genuinely solid, say so plainly and note only real gaps — do "
    "not invent problems. Output a short bulleted critique only."
)

_VALIDATE_SYSTEM = (
    "You are the validation stage of a careful engineering assistant. Given the "
    "extracted requirements/success criteria and the FINAL answer that was given, "
    "produce an honest completion checklist. For each requirement and success "
    "criterion, mark ✓ (met), ⚠ (partially met), or ✗ (not met) with a few words. "
    "Then add one line: 'Confidence: High|Medium|Low — <one clause why>'. "
    "Be truthful; do not rubber-stamp. Output only the checklist."
)


async def _gen(url: str, model: str, messages: List[Dict], *,
               headers: Optional[Dict], temperature: float, max_tokens: int,
               timeout: int) -> str:
    """Single non-streamed stage generation. Returns '' on any failure."""
    from src.llm_core import llm_call_async
    try:
        out = await llm_call_async(
            url, model, messages,
            headers=headers, temperature=temperature,
            max_tokens=max_tokens, timeout=timeout,
        )
        return _strip(out)
    except Exception as e:
        logger.warning("process_pipeline stage failed: %s", e)
        return ""


def _strip(text: str) -> str:
    """Drop reasoning-model <think> scratchpads from a stage output."""
    try:
        from src.research_utils import strip_thinking
        return (strip_thinking(text) or text).strip()
    except Exception:
        return (text or "").strip()


def _last_user_text(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


def _augment_with_requirements(messages: List[Dict], requirements: str) -> List[Dict]:
    """Inject requirements as a system directive guiding the single answer (light)."""
    if not requirements:
        return messages
    directive = {
        "role": "system",
        "content": (
            "[Process discipline] You have already analyzed the request:\n\n"
            f"{requirements}\n\n"
            "Write your answer so it satisfies every requirement and success "
            "criterion above. Do not restate this analysis; just produce the answer."
        ),
    }
    return list(messages) + [directive]


def _augment_for_revise(messages: List[Dict], requirements: str,
                        draft: str, critique: str) -> List[Dict]:
    """Build messages whose generation is the FINAL revised answer (full)."""
    directive = {
        "role": "system",
        "content": (
            "[Process discipline] You have already worked this through internally:\n\n"
            + (f"{requirements}\n\n" if requirements else "")
            + "YOUR DRAFT:\n" + (draft or "(none)") + "\n\n"
            + "SELF-CRITIQUE OF THE DRAFT:\n" + (critique or "(none)") + "\n\n"
            "Now produce the FINAL answer for the user: resolve every issue raised "
            "in the critique and satisfy all requirements and success criteria. "
            "Output only the final answer — no mention of the draft or critique."
        ),
    }
    return list(messages) + [directive]


# ---- public API -------------------------------------------------------------
async def prepare(url: str, model: str, messages: List[Dict], *, level: str,
                  headers: Optional[Dict] = None, temperature: float = 0.7,
                  timeout: int = 120) -> PipelinePrep:
    """Run pre-stream stages. Returns augmented messages + trace for the final gen.

    On ``level == 'none'`` (or any failure) returns the original messages so the
    caller streams exactly as before.
    """
    if level not in ("light", "full"):
        return PipelinePrep(messages=messages, level="none")

    try:
        user_text = _last_user_text(messages)

        # Stage 1 — requirement extraction (light + full).
        requirements = await _gen(
            url, model,
            [{"role": "system", "content": _REQUIREMENTS_SYSTEM},
             {"role": "user", "content": user_text}],
            headers=headers, temperature=_ANALYSIS_TEMPERATURE,
            max_tokens=_REQUIREMENTS_MAX_TOKENS, timeout=timeout,
        )
        trace: List[TraceEvent] = []
        if requirements:
            trace.append(TraceEvent("requirements", "Requirements & assumptions", requirements))

        if level == "light":
            return PipelinePrep(
                messages=_augment_with_requirements(messages, requirements),
                trace=trace, requirements=requirements, level="light",
            )

        # Stage 2 — draft (full only).
        draft = await _gen(
            url, model, messages,
            headers=headers, temperature=temperature,
            max_tokens=_DRAFT_MAX_TOKENS, timeout=timeout,
        )
        if not draft:
            # Drafting failed — degrade to 'light' so the turn still benefits.
            return PipelinePrep(
                messages=_augment_with_requirements(messages, requirements),
                trace=trace, requirements=requirements, level="light",
            )

        # Stage 3 — self-critique (full only).
        critique = await _gen(
            url, model,
            [{"role": "system", "content": _CRITIQUE_SYSTEM},
             {"role": "user", "content": (
                 (f"Requirements:\n{requirements}\n\n" if requirements else "")
                 + f"User request:\n{user_text}\n\nDRAFT to critique:\n{draft}"
             )}],
            headers=headers, temperature=_ANALYSIS_TEMPERATURE,
            max_tokens=_CRITIQUE_MAX_TOKENS, timeout=timeout,
        )
        if critique:
            trace.append(TraceEvent("critique", "Self-critique", critique))

        return PipelinePrep(
            messages=_augment_for_revise(messages, requirements, draft, critique),
            trace=trace, requirements=requirements, level="full",
        )
    except Exception as e:  # never break the turn
        logger.warning("process_pipeline.prepare failed, falling back to plain: %s", e)
        return PipelinePrep(messages=messages, level="none")


async def validate(url: str, model: str, *, requirements: str, user_text: str,
                   final_answer: str, headers: Optional[Dict] = None,
                   timeout: int = 120) -> Optional[TraceEvent]:
    """Completion checklist over the final answer. None on failure/empty."""
    if not (final_answer or "").strip():
        return None
    try:
        checklist = await _gen(
            url, model,
            [{"role": "system", "content": _VALIDATE_SYSTEM},
             {"role": "user", "content": (
                 (f"Requirements & success criteria:\n{requirements}\n\n" if requirements
                  else f"User request:\n{user_text}\n\n")
                 + f"FINAL answer given:\n{final_answer}"
             )}],
            headers=headers, temperature=_ANALYSIS_TEMPERATURE,
            max_tokens=_VALIDATE_MAX_TOKENS, timeout=timeout,
        )
        if checklist:
            return TraceEvent("validation", "Completion checklist", checklist)
    except Exception as e:
        logger.warning("process_pipeline.validate failed: %s", e)
    return None


async def run(url: str, model: str, messages: List[Dict], *, level: str,
              headers: Optional[Dict] = None, temperature: float = 0.7,
              max_tokens: int = 0, timeout: int = 120) -> Dict:
    """All-in-one non-streaming pipeline for /api/chat.

    Returns ``{"answer": str, "trace": List[dict], "level": str}``. On any
    failure returns a plain single generation so the endpoint never breaks.
    """
    from src.llm_core import llm_call_async

    if level not in ("light", "full"):
        answer = await llm_call_async(url, model, messages, headers=headers,
                                      temperature=temperature, max_tokens=max_tokens,
                                      timeout=timeout)
        return {"answer": answer, "trace": [], "level": "none"}

    try:
        prep = await prepare(url, model, messages, level=level, headers=headers,
                             temperature=temperature, timeout=timeout)
        answer = await llm_call_async(url, model, prep.messages, headers=headers,
                                      temperature=temperature, max_tokens=max_tokens,
                                      timeout=timeout)
        trace = [t.as_dict() for t in prep.trace]
        val = await validate(url, model, requirements=prep.requirements,
                            user_text=_last_user_text(messages),
                            final_answer=_strip(answer), headers=headers, timeout=timeout)
        if val:
            trace.append(val.as_dict())
        return {"answer": answer, "trace": trace, "level": prep.level}
    except Exception as e:
        logger.warning("process_pipeline.run failed, plain fallback: %s", e)
        answer = await llm_call_async(url, model, messages, headers=headers,
                                      temperature=temperature, max_tokens=max_tokens,
                                      timeout=timeout)
        return {"answer": answer, "trace": [], "level": "none"}
