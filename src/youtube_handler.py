"""
YouTube handling — transcript extraction, comment fetching (yt-dlp),
and context formatting for LLM injection. Used by chat_handler.py.
"""

import asyncio
import glob
import html
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YOUTUBE_INSTRUCTION_PROMPT = """When the user shares a YouTube video, respond with a structured breakdown:

1. **Summary** — Concise overview of the video's content and main thesis (2-4 sentences)
2. **Key Points** — Bullet list of the most important topics, arguments, or moments
3. **Notable Timestamps** — If timestamps are available from the transcript, highlight 3-5 interesting moments with their approximate timestamps (e.g. "03:45 — discusses X")
4. **Audience Reception** — If comments are available, summarize what viewers think: general sentiment, top reactions, any debate or controversy

Keep it conversational and concise. Do NOT web search for this video — use only the transcript and comments provided."""

# ---------------------------------------------------------------------------
# Init / helpers
# ---------------------------------------------------------------------------

# Will be set at startup by init_youtube()
YouTubeTranscriptApi = None
YOUTUBE_AVAILABLE = False


def _find_ytdlp() -> str:
    """Find the yt-dlp binary: venv bin first, then system PATH."""
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which("yt-dlp")
    return found or "yt-dlp"


def init_youtube():
    """Import and cache the YouTube transcript API."""
    global YouTubeTranscriptApi, YOUTUBE_AVAILABLE
    try:
        from youtube_transcript_api import YouTubeTranscriptApi as _Api
        YouTubeTranscriptApi = _Api
        YOUTUBE_AVAILABLE = True
        logger.info("YouTube transcript API available")
    except ImportError as e:
        logger.warning(f"youtube-transcript-api not installed: {e}")
        YOUTUBE_AVAILABLE = False


def is_youtube_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    return "youtube.com" in url or "youtu.be" in url


def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    if not isinstance(url, str):
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            params = urllib.parse.parse_qs(parsed.query)
            if "v" in params:
                return params["v"][0]
        elif parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[-1]
    elif parsed.hostname == "youtu.be":
        return parsed.path[1:]
    return None


def _fmt_ts(seconds: float) -> str:
    """Format seconds as H:MM:SS for long videos, MM:SS for short ones."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def fetch_metadata_oembed(url_or_id: str, timeout: int = 10) -> Dict[str, Any]:
    """Fetch video title/channel/thumbnail via YouTube's oEmbed endpoint.

    oEmbed needs no API key and no proof-of-origin token, so it stays reliable
    where the watch-page scrape and the timedtext endpoint do not. Best-effort:
    returns {} on any failure.
    """
    vid = extract_youtube_id(url_or_id) or (url_or_id if "/" not in url_or_id else "")
    target = url_or_id if "youtu" in url_or_id else f"https://www.youtube.com/watch?v={vid}"
    try:
        import httpx

        r = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": target, "format": "json"},
            timeout=timeout,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return {}
        d = r.json()
        return {
            "title": d.get("title", ""),
            "channel": d.get("author_name", ""),
            "channel_url": d.get("author_url", ""),
            "thumbnail": d.get("thumbnail_url", ""),
        }
    except Exception as e:  # never fatal
        logger.debug("oEmbed metadata failed for %s: %s", target, e)
        return {}


def _parse_vtt(vtt_text: str):
    """Parse a WebVTT subtitle blob into (full_text, segments).

    Auto-generated captions arrive as overlapping "rolling" cues with inline
    word-timing tags (``<00:00:01.234>``). We strip the inline tags, drop exact
    repeats, then merge cues by their longest word-overlap so the running text
    reads once rather than echoing every line. ``segments`` buckets the text into
    ~30s windows so callers can still cite approximate timestamps.
    """
    ts_re = re.compile(r"(\d\d):(\d\d):(\d\d)\.\d{3}\s+-->")
    cues = []  # (start_sec, text)
    cur = None
    buf: list = []

    def _flush():
        nonlocal cur, buf
        if cur is not None and buf:
            cues.append((cur, " ".join(buf)))
        buf = []

    for ln in vtt_text.splitlines():
        m = ts_re.match(ln)
        if m:
            _flush()
            cur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            continue
        if "-->" in ln or not ln.strip() or ln.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        t = re.sub(r"<[^>]+>", "", ln)  # strip inline word-timing / styling tags
        t = html.unescape(t)            # &nbsp;/&amp;/&#39; -> real chars
        t = re.sub(r"\s+", " ", t).strip()  # collapses NBSP (\xa0) too
        if t:
            buf.append(t)
    _flush()

    # Drop consecutive exact duplicates (common in rolling auto-captions).
    deduped = []
    for start, txt in cues:
        if deduped and txt == deduped[-1][1]:
            continue
        deduped.append((start, txt))

    BUCKET = 30  # seconds per timestamped segment
    merged = ""
    segments: list = []
    bucket_idx = None
    bucket_start = 0
    bucket_words: list = []

    def _push_bucket():
        if bucket_words:
            segments.append({
                "text": " ".join(bucket_words).strip(),
                "start": float(bucket_start),
                "duration": 0.0,
                "timestamp": _fmt_ts(bucket_start),
            })

    for start, txt in deduped:
        if not merged:
            novel = txt
            merged = txt
        else:
            mw = merged.split()
            tw = txt.split()
            ov = 0
            for k in range(min(len(mw), len(tw), 20), 0, -1):
                if mw[-k:] == tw[:k]:
                    ov = k
                    break
            novel = " ".join(tw[ov:])
            if novel:
                merged += " " + novel
        bi = start // BUCKET
        if bucket_idx is None:
            bucket_idx = bi
            bucket_start = start
        elif bi != bucket_idx:
            _push_bucket()
            bucket_words = []
            bucket_idx = bi
            bucket_start = start
        if novel:
            bucket_words.append(novel)
    _push_bucket()

    return merged.strip(), segments


def extract_transcript_ytdlp_sync(
    video_id: str, languages=("en",), timeout: int = 90
) -> Dict[str, Any]:
    """Extract a transcript via yt-dlp subtitles (the path that survives YouTube's
    timedtext gating). Downloads manual + auto English subs as VTT, parses, dedupes.
    """
    ytdlp = _find_ytdlp()
    langs = ",".join([f"{l}.*" for l in languages] + list(languages)) or "en.*,en"
    with tempfile.TemporaryDirectory() as tmp:
        out_tmpl = str(Path(tmp) / "%(id)s")
        cmd = [
            ytdlp,
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", langs,
            "--sub-format", "vtt",
            "--no-warnings",
            "-o", out_tmpl,
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "yt-dlp transcript timed out", "transcript": None}
        except FileNotFoundError:
            return {"success": False, "error": "yt-dlp not installed", "transcript": None}

        vtts = sorted(glob.glob(str(Path(tmp) / "*.vtt")))
        if not vtts:
            err = (proc.stderr or "").strip().splitlines()
            tail = err[-1] if err else "no subtitles available"
            return {"success": False, "error": f"no subtitles ({tail[:160]})", "transcript": None}

        try:
            vtt_text = Path(vtts[0]).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"success": False, "error": f"read sub failed: {e}", "transcript": None}

    full_text, segments = _parse_vtt(vtt_text)
    if not full_text:
        return {"success": False, "error": "empty transcript after parse", "transcript": None}

    # Inline word-timing tags only appear in auto-generated captions.
    is_generated = bool(re.search(r"<\d\d:\d\d:\d\d\.\d{3}>", vtt_text))
    return {
        "success": True,
        "transcript": full_text,
        "video_id": video_id,
        "language": languages[0] if languages else "en",
        "is_generated": is_generated,
        "segments": segments,
        "engine": "yt-dlp",
    }


def extract_transcript_sync(url: str, video_id: str, ytdlp_timeout: int = 90) -> Dict[str, Any]:
    """Best-effort transcript: try youtube-transcript-api (cheap when it works),
    then fall back to yt-dlp subtitles (robust against POT-token gating)."""
    if YOUTUBE_AVAILABLE and YouTubeTranscriptApi is not None:
        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id)
            formatted = []
            for snippet in transcript:
                text = snippet.text.strip()
                if not text:
                    continue
                start = snippet.start
                formatted.append({
                    "text": text,
                    "start": start,
                    "duration": snippet.duration,
                    "timestamp": _fmt_ts(start),
                })
            if formatted:
                return {
                    "success": True,
                    "transcript": " ".join(e["text"] for e in formatted),
                    "video_id": video_id,
                    "language": "en",
                    "is_generated": False,
                    "segments": formatted,
                    "engine": "youtube-transcript-api",
                }
        except Exception as e:
            logger.info("transcript-api failed for %s (%s); falling back to yt-dlp", video_id, e)

    return extract_transcript_ytdlp_sync(video_id, timeout=ytdlp_timeout)


async def extract_transcript_async(
    url: str, video_id: str, max_retries: int = 3
) -> Dict[str, Any]:
    """Async wrapper around :func:`extract_transcript_sync` (runs the blocking
    API/yt-dlp work in a thread so the event loop is never stalled)."""
    try:
        return await asyncio.to_thread(extract_transcript_sync, url, video_id)
    except Exception as e:
        logger.warning("transcript extraction failed for %s: %s", video_id, e)
        return {"success": False, "error": str(e), "transcript": None}


def format_transcript_for_context(
    transcript_data: Dict[str, Any], url: str,
    title: str = "", channel: str = ""
) -> str:
    """Format transcript data for inclusion in LLM context."""
    if not transcript_data.get("success"):
        header = ""
        if title:
            header = f" \"{title}\""
            if channel:
                header += f" by {channel}"
        return f"\n[YouTube Video{header}: Transcript unavailable ({transcript_data.get('error', 'Unknown error')}). Use the comments below if available, do NOT web search for this video.]"

    transcript = transcript_data.get("transcript", "")
    video_id = transcript_data.get("video_id", "")
    language = transcript_data.get("language", "unknown")
    is_generated = transcript_data.get("is_generated", False)
    segments = transcript_data.get("segments", [])

    ctx = "\n[YOUTUBE VIDEO TRANSCRIPT]\n"
    if title:
        ctx += f"Title: {title}\n"
    if channel:
        ctx += f"Channel: {channel}\n"
    ctx += f"Video ID: {video_id}\n"
    ctx += f"Language: {language}\n"
    ctx += f"Source: {'Auto-generated' if is_generated else 'Manual'}\n"
    ctx += f"URL: {url}\n\n"
    # Include timestamped segments for the LLM to reference
    if segments:
        ctx += "Timestamped Transcript:\n"
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            ctx += f"[{seg['timestamp']}] {seg['text']}\n"
        # Check length — fall back to plain text if too long
        if len(ctx) > 12000:
            ctx = ctx[:ctx.index("Timestamped Transcript:\n")]
            ctx += "Transcript:\n"
            ctx += transcript
    else:
        ctx += "Transcript:\n"
        ctx += transcript
    ctx += "\n[END TRANSCRIPT]\n"
    return ctx


async def fetch_youtube_comments(
    video_id: str, max_comments: int = 25, timeout: int = 30
) -> Dict[str, Any]:
    """Fetch top comments for a YouTube video using yt-dlp.

    Returns dict with 'success', 'comments' list, 'error'.
    """
    try:
        cmd = [
            _find_ytdlp(),
            "--skip-download",
            "--write-comments",
            "--extractor-args", f"youtube:max_comments={max_comments},all,100,0",
            "--dump-json",
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            f"https://www.youtube.com/watch?v={video_id}",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Bound the wait on the process actually finishing, not on spawning it.
        # create_subprocess_exec returns as soon as the child starts, so wrapping
        # it in wait_for never enforces the timeout — proc.communicate() is the
        # blocking step. Kill and reap the child if it overruns so it does not
        # linger after we return.
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            return {"success": False, "error": f"yt-dlp failed: {stderr.decode()[:200]}", "comments": []}

        data = json.loads(stdout.decode())
        title = data.get("title", "")
        channel = data.get("channel", "") or data.get("uploader", "")
        raw_comments = data.get("comments", [])

        comments = []
        for c in raw_comments[:max_comments]:
            if not isinstance(c, dict):
                continue
            text = (c.get("text") or "").strip()
            if not text:
                continue
            comments.append({
                "author": c.get("author", "Unknown"),
                "text": text,
                "likes": c.get("like_count", 0),
            })

        # Sort by likes descending — most popular comments first
        comments.sort(key=lambda x: x.get("likes", 0), reverse=True)

        return {"success": True, "comments": comments, "count": len(comments),
                "title": title, "channel": channel}

    except asyncio.TimeoutError:
        logger.warning(f"Comment fetch timed out for {video_id}")
        return {"success": False, "error": "Comment fetch timed out", "comments": []}
    except FileNotFoundError:
        logger.warning("yt-dlp not installed — cannot fetch comments")
        return {"success": False, "error": "yt-dlp not installed", "comments": []}
    except Exception as e:
        logger.warning(f"Failed to fetch comments for {video_id}: {e}")
        return {"success": False, "error": str(e), "comments": []}


def format_comments_for_context(comments_data: Dict[str, Any], url: str) -> str:
    """Format YouTube comments for inclusion in LLM context."""
    if not comments_data.get("success") or not comments_data.get("comments"):
        return ""

    comments = comments_data["comments"]
    ctx = f"\n[YOUTUBE VIDEO COMMENTS — Top {len(comments)} by popularity]\n"
    ctx += f"URL: {url}\n\n"

    for i, c in enumerate(comments, 1):
        likes = c.get("likes", 0)
        likes_str = f" [{likes} likes]" if likes else ""
        ctx += f"{i}. @{c['author']}{likes_str}: {c['text']}\n\n"

    if len(ctx) > 4000:
        ctx = ctx[:4000] + "\n[Comments truncated]\n"

    ctx += "[END COMMENTS]\n"
    return ctx
