"""Per-turn process-discipline router.

Decides how much "Claude-like process discipline" a given turn deserves, so the
expensive multi-stage pipeline (src/process_pipeline.py) only runs when it pays
off. On a local 30B every stage is another full generation, so casual chat must
stay a single call while code/architecture/research turns get the full
Reason→Draft→Critique→Revise→Validate treatment.

UNIVERSAL: this is pure heuristic text classification — no model call, no
model-specific assumptions. It works identically for Qwen, hermes3, gpt-oss, or
any remote endpoint.

Levels (ascending cost):
    "none"  — single normal generation, no extra stages.
    "light" — requirement extraction + completion validation around the answer.
    "full"  — requirements + draft + self-critique + revise + validation.

The admin ``process_level`` setting overrides the router:
    "off"   — always "none" (feature dormant; Push-1 default).
    "auto"  — use this router's per-turn decision.
    "light" — force at least "light" on substantive turns.
    "full"  — force "full" on substantive turns.
(Trivial/greeting turns are always "none" regardless, so "hi" never costs 5x.)
"""
from __future__ import annotations

import re
from typing import Optional

# Strong signals that a turn is engineering / design / analysis work — the place
# the full pipeline earns its latency.
_FULL_PATTERNS = (
    r"\b(implement|build|write|create|refactor|debug|fix|optimi[sz]e|design|architect"
    r"|migrat|integrat|deploy|configure|set up|scaffold)\b",
    r"\b(architecture|algorithm|data ?model|schema|API|endpoint|pipeline|concurrency"
    r"|race condition|deadlock|memory leak|performance|throughput|latency|security"
    r"|vulnerabilit|auth(entication|orization)?|encryption|threat model)\b",
    r"\b(unit test|integration test|test coverage|edge case|production|rollback|CI/CD)\b",
    r"\b(why (does|is|are|won'?t|doesn'?t)|root cause|trade-?off|pros and cons"
    r"|compare .* (vs|versus|against)|which (approach|option|design))\b",
    r"```",                       # a code fence in the message
    r"(?m)^\s*(def |class |function |import |#include|SELECT |CREATE TABLE)",
    r"\b[\w./-]+\.(py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|sql|yaml|yml|sh|json)\b",  # filename
)

# Substantive but not necessarily engineering — explanation / advice / planning.
_LIGHT_PATTERNS = (
    r"\b(explain|how (do|does|can|should|would)|walk me through|step by step"
    r"|what(?:'s| is| are) the (best|right|difference)|should i|help me (understand|plan|decide)"
    r"|summari[sz]e|analy[sz]e|evaluate|review|critique|plan|outline|draft)\b",
    r"\?",                         # contains a question
)

# Trivial turns that must never trigger the pipeline.
_TRIVIAL_PATTERNS = (
    r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|ty|thx|ok|okay|k|cool|nice|great"
    r"|lol|got it|gotcha|yep|yeah|no|nope|sure|please|continue|go on|👍|👌)\s*[!.…]*\s*$",
)

_LEVELS = ("none", "light", "full")


def _max_level(a: str, b: str) -> str:
    return a if _LEVELS.index(a) >= _LEVELS.index(b) else b


def classify_turn(message: str, *, agent_mode: bool = False) -> str:
    """Heuristic per-turn level: 'none' | 'light' | 'full'. No model call."""
    text = (message or "").strip()
    if not text:
        return "none"

    low = text.lower()

    # Trivial greetings/acks: always cheap, even mid-engineering session.
    for pat in _TRIVIAL_PATTERNS:
        if re.match(pat, low, re.IGNORECASE):
            return "none"

    for pat in _FULL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "full"

    level = "none"
    for pat in _LIGHT_PATTERNS:
        if re.search(pat, low, re.IGNORECASE):
            level = "light"
            break

    # Long, dense prompts are usually substantive even without a keyword hit.
    if level == "none" and len(text) > 600:
        level = "light"

    # Agent-mode turns lean toward real work; nudge borderline 'light' to engage.
    if agent_mode and level == "light" and len(text) > 200:
        level = "full"

    return level


def resolve_level(message: str, setting: Optional[str], *, agent_mode: bool = False) -> str:
    """Combine the admin ``process_level`` setting with the per-turn classifier.

    Returns the final level to run. ``setting`` is the raw ``process_level``
    value ("off"|"auto"|"light"|"full"; None/unknown treated as "off").
    """
    mode = (setting or "off").strip().lower()
    if mode == "off":
        return "none"

    auto = classify_turn(message, agent_mode=agent_mode)
    # Trivial turns stay cheap no matter what the admin forced.
    if auto == "none":
        return "none"

    if mode == "auto":
        return auto
    if mode == "light":
        return _max_level(auto, "light") if auto != "full" else "full"
    if mode == "full":
        return "full"
    return auto
