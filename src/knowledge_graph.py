"""
knowledge_graph.py

A lightweight, owner-scoped entity knowledge graph that augments vector RAG.

Design goals (all in service of "add signal without breaking or slowing anything"):

  * Extraction happens at INGEST time, off the request path, in a background
    thread — it never blocks an upload finishing and never touches a chat turn.
  * Query-time use is a single cheap SQLite read plus a bounded Chroma get; it is
    gated, capped, and wrapped so a failure degrades to "no expansion", never an
    error.
  * Storage is a standalone SQLite file under data/ — it does not touch app.db,
    its schema, or any migration. Deleting the file simply empties the graph.

The graph is two relations over entity strings:

    mentions(owner, entity, doc_id)   -- entity appears in this RAG chunk
    edges(owner, src, dst, weight)    -- src and dst co-occur / are related

At query time we take the doc_ids of the strongest first-stage hits, look up the
entities mentioned there, hop one step out across `edges`, then surface other
doc_ids that mention those neighbours. Those extra chunks join the candidate pool
*before* reranking, so the cross-encoder still gets the final say on order.
"""

import os
import json
import logging
import sqlite3
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from src.constants import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(DATA_DIR, "knowledge_graph.db")

# Bounds — deliberately small so neither ingest nor query work can balloon.
MAX_ENTITIES_PER_CHUNK = 12
MAX_CHUNK_CHARS = 4000          # cap text sent to the extractor
DEFAULT_EXPAND_SEEDS = 3        # how many top hits seed the 1-hop walk
DEFAULT_EXPAND_LIMIT = 5        # max neighbour doc_ids pulled into the pool

# Per-chunk extraction is the LLM bottleneck during a big ingest/backfill. The
# utility model (e.g. hermes3:8b) serves ~1 call at a time per GPU lane, so
# firing many concurrent extracts just queues them past the request timeout and
# they get aborted — work lost, graph half-populated. Cap total concurrent
# utility-model calls across every path (enrich_async threads AND the backfill
# pool) so each call actually completes. The extract timeout is generous since
# this is best-effort background work, not a latency-sensitive chat turn.
#
# Dual-GPU: list extra utility lanes (one ollama per GPU) in KG_EXTRA_UTILITY_URLS
# as comma-separated host:port (or full URLs). Each extract is dispatched to the
# lane with the fewest in-flight calls (join-shortest-queue), so a faster GPU
# naturally absorbs more work instead of idling behind a 50/50 split. The other
# lane is still kept as a per-call fallback, so one wedged GPU degrades to
# single-lane instead of dropping work. Concurrency cap is sized 3 per lane.
EXTRACT_PER_LANE_CONCURRENCY = 3
EXTRA_UTILITY_NETLOCS = [
    s.strip() for s in os.environ.get("KG_EXTRA_UTILITY_URLS", "").split(",") if s.strip()
]
EXTRACT_MAX_CONCURRENCY = EXTRACT_PER_LANE_CONCURRENCY * (1 + len(EXTRA_UTILITY_NETLOCS))
EXTRACT_TIMEOUT_S = 120         # per-call timeout for background extraction
_extract_sem = threading.Semaphore(EXTRACT_MAX_CONCURRENCY)

# Per-lane in-flight counters for join-shortest-queue lane selection.
_lane_lock = threading.Lock()
_lane_inflight: List[int] = []


def _swap_netloc(url: str, netloc: str) -> str:
    """Return ``url`` with its host:port replaced by ``netloc``.

    ``netloc`` may be a bare ``host:port`` or a full ``scheme://host:port`` URL;
    in the latter case its netloc is used. Preserves the original path so the
    OpenAI-vs-Ollama chat path stays correct across lanes.
    """
    from urllib.parse import urlparse, urlunparse
    if "://" in netloc:
        netloc = urlparse(netloc).netloc
    return urlunparse(urlparse(url)._replace(netloc=netloc))


def _utility_lanes(primary):
    """Pick the per-call lane order by join-shortest-queue.

    ``primary`` is the resolved (url, model, headers). Extra lanes reuse the same
    model+headers with the host:port swapped to each KG_EXTRA_UTILITY_URLS entry.
    Returns ``(ordered_lanes, lane_idx)`` where the least-loaded lane leads (and
    is charged one in-flight slot); the caller must release it via
    ``_release_lane(lane_idx)`` once the call completes. ``lane_idx`` is None when
    there is only one lane (nothing to balance).
    """
    url, model, headers = primary
    lanes = [primary]
    for netloc in EXTRA_UTILITY_NETLOCS:
        try:
            lanes.append((_swap_netloc(url, netloc), model, headers))
        except Exception:
            pass
    if len(lanes) <= 1:
        return lanes, None
    with _lane_lock:
        while len(_lane_inflight) < len(lanes):
            _lane_inflight.append(0)
        idx = min(range(len(lanes)), key=lambda i: _lane_inflight[i])
        _lane_inflight[idx] += 1
    ordered = lanes[idx:] + lanes[:idx]
    return ordered, idx


def _release_lane(lane_idx) -> None:
    """Release the in-flight slot charged by ``_utility_lanes``."""
    if lane_idx is None:
        return
    with _lane_lock:
        if 0 <= lane_idx < len(_lane_inflight) and _lane_inflight[lane_idx] > 0:
            _lane_inflight[lane_idx] -= 1

_conn: Optional[sqlite3.Connection] = None
_conn_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Settings shims (kept thin so tests can monkeypatch cleanly)
# ---------------------------------------------------------------------------

def _get_setting(key: str, default):
    try:
        from src.settings import get_setting
        return get_setting(key, default)
    except Exception:
        return default


def is_enabled() -> bool:
    """Master gate for the whole feature (extraction + expansion)."""
    return bool(_get_setting("kg_enabled", True))


def expansion_enabled() -> bool:
    """Query-time expansion can be turned off independently of extraction."""
    return is_enabled() and bool(_get_setting("kg_expand_on_query", True))


def expand_limit() -> int:
    try:
        v = int(_get_setting("kg_expand_limit", DEFAULT_EXPAND_LIMIT))
        return v if v > 0 else DEFAULT_EXPAND_LIMIT
    except Exception:
        return DEFAULT_EXPAND_LIMIT


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _connect() -> Optional[sqlite3.Connection]:
    """Return a shared, lazily-initialised connection (or None if unusable)."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            return _conn
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mentions (
                    owner  TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    PRIMARY KEY (owner, entity, doc_id)
                );
                CREATE INDEX IF NOT EXISTS idx_mentions_entity
                    ON mentions(owner, entity);
                CREATE INDEX IF NOT EXISTS idx_mentions_doc
                    ON mentions(owner, doc_id);

                CREATE TABLE IF NOT EXISTS edges (
                    owner  TEXT NOT NULL,
                    src    TEXT NOT NULL,
                    dst    TEXT NOT NULL,
                    weight INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (owner, src, dst)
                );
                CREATE INDEX IF NOT EXISTS idx_edges_src
                    ON edges(owner, src);
                """
            )
            conn.commit()
            _conn = conn
            return _conn
        except Exception as exc:
            logger.warning("knowledge_graph: DB unavailable (%s)", exc)
            return None


def reset_for_tests(path: Optional[str] = None) -> None:
    """Drop the cached connection (and optionally repoint the DB path)."""
    global _conn, DB_PATH
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None
    if path is not None:
        DB_PATH = path


def _norm(entity: str) -> str:
    return " ".join(str(entity).split()).strip()


def store(owner: str, doc_id: str, entities: Sequence[str],
          relations: Sequence[Tuple[str, str]]) -> None:
    """Persist one chunk's entities + relations. Best-effort, owner-scoped."""
    conn = _connect()
    if conn is None:
        return
    ents = []
    seen = set()
    for e in entities:
        n = _norm(e)
        low = n.lower()
        if n and low not in seen:
            seen.add(low)
            ents.append(n)
        if len(ents) >= MAX_ENTITIES_PER_CHUNK:
            break
    if not ents:
        return
    try:
        with _conn_lock:
            conn.executemany(
                "INSERT OR IGNORE INTO mentions(owner, entity, doc_id) VALUES (?,?,?)",
                [(owner, e.lower(), doc_id) for e in ents],
            )
            for a, b in relations:
                na, nb = _norm(a).lower(), _norm(b).lower()
                if not na or not nb or na == nb:
                    continue
                for s, d in ((na, nb), (nb, na)):  # store undirected as two rows
                    conn.execute(
                        "INSERT INTO edges(owner, src, dst, weight) VALUES (?,?,?,1) "
                        "ON CONFLICT(owner, src, dst) DO UPDATE SET weight = weight + 1",
                        (owner, s, d),
                    )
            conn.commit()
    except Exception as exc:
        logger.debug("knowledge_graph.store failed: %s", exc)


# ---------------------------------------------------------------------------
# Extraction (LLM, ingest-time only)
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You extract a knowledge graph from a text chunk. Return STRICT JSON only, "
    "no prose, no code fences. Schema: "
    '{"entities": ["..."], "relations": [["entity_a", "entity_b"], ...]}. '
    "Entities are the salient proper nouns / concepts (max 12). Relations are "
    "pairs of entities that are directly related in the text. If nothing "
    'meaningful is present, return {"entities": [], "relations": []}.'
)


def _parse_extraction(raw: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    if not raw:
        return [], []
    s = raw.strip()
    # tolerate ```json fences and leading/trailing chatter
    if "```" in s:
        s = s.split("```")[1] if s.count("```") >= 2 else s.replace("```", "")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    try:
        data = json.loads(s)
    except Exception:
        return [], []
    ents = [str(e) for e in data.get("entities", []) if str(e).strip()]
    rels: List[Tuple[str, str]] = []
    for r in data.get("relations", []):
        if isinstance(r, (list, tuple)) and len(r) >= 2 and str(r[0]).strip() and str(r[1]).strip():
            rels.append((str(r[0]), str(r[1])))
    return ents, rels


def extract(text: str, owner: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Run the utility model over one chunk. Returns ([],[]) on any failure."""
    if not text or not text.strip():
        return [], []
    try:
        from src.endpoint_resolver import (
            resolve_endpoint,
            resolve_utility_fallback_candidates,
        )
        from src.llm_core import llm_call_with_fallback

        url, model, headers = resolve_endpoint("utility", owner=owner)
        if not url or not model:
            url, model, headers = resolve_endpoint("default", owner=owner)
        if not url or not model:
            return [], []
        lanes, lane_idx = _utility_lanes((url, model, headers))
        candidates = lanes + resolve_utility_fallback_candidates(owner=owner)

        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": text[:MAX_CHUNK_CHARS]},
        ]
        try:
            with _extract_sem:
                raw = llm_call_with_fallback(
                    candidates, messages, temperature=0.0,
                    max_tokens=400, timeout=EXTRACT_TIMEOUT_S,
                )
        finally:
            _release_lane(lane_idx)
        return _parse_extraction(raw)
    except Exception as exc:
        logger.debug("knowledge_graph.extract failed: %s", exc)
        return [], []


def enrich_documents(docs: Sequence[Tuple[str, str]], owner: str) -> int:
    """Extract + store for a list of (doc_id, text). Returns chunks enriched."""
    if not is_enabled():
        return 0
    enriched = 0
    for doc_id, text in docs:
        ents, rels = extract(text, owner)
        if ents:
            store(owner, doc_id, ents, rels)
            enriched += 1
    return enriched


def existing_doc_ids(owner: str) -> set:
    """Doc ids already enriched for ``owner`` so a backfill can skip them.

    Makes the bulk backfill idempotent + resumable: re-running after an
    interrupt only processes chunks that never made it into the graph.
    """
    conn = _connect()
    if conn is None:
        return set()
    try:
        with _conn_lock:
            rows = conn.execute(
                "SELECT DISTINCT doc_id FROM mentions WHERE owner = ?", (owner,)
            ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def enrich_async(docs: Sequence[Tuple[str, str]], owner: str) -> None:
    """Fire-and-forget background enrichment so ingest never waits on the LLM."""
    if not is_enabled() or not docs:
        return
    payload = list(docs)

    def _run():
        try:
            n = enrich_documents(payload, owner)
            if n:
                logger.info("knowledge_graph: enriched %s chunk(s) for %s", n, owner or "-")
        except Exception as exc:
            logger.debug("knowledge_graph.enrich_async failed: %s", exc)

    threading.Thread(target=_run, name="kg-enrich", daemon=True).start()


# ---------------------------------------------------------------------------
# Query-time expansion (cheap, bounded SQLite reads)
# ---------------------------------------------------------------------------

def neighbor_doc_ids(seed_doc_ids: Sequence[str], owner: str,
                     exclude: Optional[set] = None,
                     limit: int = DEFAULT_EXPAND_LIMIT) -> List[str]:
    """Given seed chunk ids, return up to `limit` related chunk ids via 1 hop.

    seeds -> entities mentioned in seeds -> 1-hop neighbour entities ->
    other chunks mentioning those neighbours. All owner-scoped; returns [] on
    any problem.
    """
    if not seed_doc_ids:
        return []
    conn = _connect()
    if conn is None:
        return []
    exclude = set(exclude or set())
    exclude.update(seed_doc_ids)
    try:
        with _conn_lock:
            seed_q = ",".join("?" for _ in seed_doc_ids)
            seed_entities = {
                row[0] for row in conn.execute(
                    f"SELECT DISTINCT entity FROM mentions "
                    f"WHERE owner=? AND doc_id IN ({seed_q})",
                    (owner, *seed_doc_ids),
                ).fetchall()
            }
            if not seed_entities:
                return []

            ent_q = ",".join("?" for _ in seed_entities)
            neighbours = {
                row[0] for row in conn.execute(
                    f"SELECT dst FROM edges WHERE owner=? AND src IN ({ent_q}) "
                    f"ORDER BY weight DESC LIMIT 50",
                    (owner, *seed_entities),
                ).fetchall()
            }
            # The seed entities themselves are valid retrieval anchors too.
            anchors = neighbours | seed_entities
            if not anchors:
                return []

            anc_q = ",".join("?" for _ in anchors)
            rows = conn.execute(
                f"SELECT doc_id, COUNT(*) AS hits FROM mentions "
                f"WHERE owner=? AND entity IN ({anc_q}) "
                f"GROUP BY doc_id ORDER BY hits DESC",
                (owner, *anchors),
            ).fetchall()

        out: List[str] = []
        for doc_id, _hits in rows:
            if doc_id in exclude:
                continue
            out.append(doc_id)
            if len(out) >= limit:
                break
        return out
    except Exception as exc:
        logger.debug("knowledge_graph.neighbor_doc_ids failed: %s", exc)
        return []
