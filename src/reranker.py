"""
reranker.py

Cross-encoder reranking for RAG retrieval.

After the hybrid vector+keyword stage produces a candidate pool, a cross-encoder
re-scores each (query, document) pair *jointly* — far more accurate than the
bi-encoder cosine similarity used for the first-stage recall, at the cost of
running the model once per candidate. The standard recipe is therefore:
over-fetch a wide pool cheaply, then rerank down to the final top-k.

Backend: fastembed's local ONNX ``TextCrossEncoder`` (no new dependency — the
project already ships fastembed for embeddings, and the model cache lives under
the same ``data/fastembed_cache`` directory). Mirrors embeddings.py:
  * lazy load on first use,
  * a process-level "down" latch so a missing/broken model doesn't pay the load
    cost on every single retrieval, and
  * a graceful no-op fallback (return the input order) so reranking can never
    take RAG down — a failure just degrades to the first-stage ranking.

Enable/disable and model choice come from settings (see DEFAULT_SETTINGS):
  rag_rerank_enabled   (bool, default True)
  rag_rerank_model     (str,  default Xenova/ms-marco-MiniLM-L-6-v2 — small/fast)
  rag_rerank_overfetch (int,  default 5 — candidate pool = k * overfetch)
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.constants import FASTEMBED_CACHE_DIR

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_OVERFETCH = 5

_encoder = None            # cached TextCrossEncoder instance
_encoder_model = None      # model name the cached encoder was built with
_reranker_down = False     # process latch: skip reload of a known-broken model


def _get_setting(key: str, default: Any) -> Any:
    """Read a setting without hard-coupling import order at module load."""
    try:
        from src.settings import get_setting

        val = get_setting(key, default)
        return default if val is None else val
    except Exception:
        return default


def is_enabled() -> bool:
    return bool(_get_setting("rag_rerank_enabled", True))


def overfetch_multiplier() -> int:
    try:
        n = int(_get_setting("rag_rerank_overfetch", DEFAULT_OVERFETCH))
        return n if n >= 1 else DEFAULT_OVERFETCH
    except (TypeError, ValueError):
        return DEFAULT_OVERFETCH


def reset_reranker_state() -> None:
    """Clear the 'reranker is down' latch and drop the cached encoder so the
    next call re-probes. Call this when the rerank model setting changes."""
    global _encoder, _encoder_model, _reranker_down
    _encoder = None
    _encoder_model = None
    _reranker_down = False


def _get_encoder():
    """Lazily build (and cache) the cross-encoder. Returns None if unavailable;
    a None return is the no-op signal for callers to keep first-stage order."""
    global _encoder, _encoder_model, _reranker_down

    model_name = str(_get_setting("rag_rerank_model", DEFAULT_RERANK_MODEL)) or DEFAULT_RERANK_MODEL

    # Rebuild if the configured model changed under us.
    if _encoder is not None and _encoder_model == model_name:
        return _encoder
    if _encoder is not None and _encoder_model != model_name:
        _encoder = None
        _encoder_model = None
        _reranker_down = False

    if _reranker_down:
        return None

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        cache_dir = FASTEMBED_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        _encoder = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
        _encoder_model = model_name
        logger.info("Reranker loaded model=%s", model_name)
        return _encoder
    except Exception as e:
        _reranker_down = True
        logger.warning(
            "Cross-encoder reranker unavailable (%s); RAG falls back to "
            "first-stage ranking for the rest of this process", e,
        )
        return None


def score(query: str, documents: Sequence[str]) -> Optional[List[float]]:
    """Return one relevance score per document, aligned with ``documents``.
    Higher is more relevant. Returns None if reranking is unavailable."""
    if not query or not documents:
        return None
    encoder = _get_encoder()
    if encoder is None:
        return None
    try:
        return [float(s) for s in encoder.rerank(query, list(documents))]
    except Exception as e:
        logger.warning("rerank scoring failed: %s", e)
        return None


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    *,
    text_key: str = "document",
    text_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    score_key: str = "rerank_score",
) -> List[Dict[str, Any]]:
    """Reorder ``candidates`` by cross-encoder relevance and return the top_k.

    Each returned candidate gets ``score_key`` set to its rerank score. If
    reranking is disabled or unavailable, the input order is preserved and the
    first ``top_k`` are returned unchanged (no score_key added) — callers can
    rely on always getting a sensible list back.

    text_key  — dict key holding the candidate text (default "document").
    text_fn   — optional override to extract text from a candidate.
    """
    if top_k <= 0 or not candidates:
        return candidates[: max(top_k, 0)]
    if not is_enabled():
        return candidates[:top_k]

    extract = text_fn or (lambda c: c.get(text_key, "") or "")
    docs = [extract(c) for c in candidates]
    scores = score(query, docs)
    if scores is None:
        return candidates[:top_k]

    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    out: List[Dict[str, Any]] = []
    for i in order[:top_k]:
        c = dict(candidates[i])
        c[score_key] = round(scores[i], 4)
        out.append(c)
    return out
