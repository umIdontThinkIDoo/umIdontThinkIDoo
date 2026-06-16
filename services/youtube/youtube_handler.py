"""Compatibility shim — the canonical YouTube handler lives in
``src.youtube_handler``.

This module used to carry its own copy of the transcript/comment logic, which
drifted from ``src/youtube_handler.py`` (the version chat_handler and the
web-search content fetcher use). To stop the two from diverging again, this file
now just re-exports the canonical implementation, mirroring the
``src/search/content.py`` -> ``services/search/content.py`` shim pattern.
"""

import sys

from src import youtube_handler as _impl

# Make ``services.youtube.youtube_handler`` *be* the canonical module so existing
# imports (``from services.youtube.youtube_handler import ...``) and
# ``services/youtube/__init__.py`` resolve every name to the same object.
sys.modules[__name__] = _impl
