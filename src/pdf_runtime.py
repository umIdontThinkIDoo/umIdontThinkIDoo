"""Small helpers for optional PDF runtime dependencies."""

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

PDF_VIEWER_PYMUPDF_MISSING = (
    "PDF viewer requires PyMuPDF. Install optional PDF dependencies with "
    "`pip install -r requirements-optional.txt` (PyMuPDF is AGPL-3.0)."
)


def load_pymupdf_for_pdf_viewer():
    """Return the PyMuPDF module, or raise a user-facing setup hint."""
    try:
        import fitz  # PyMuPDF, optional
    except ImportError as exc:
        raise RuntimeError(PDF_VIEWER_PYMUPDF_MISSING) from exc
    return fitz


def ensure_pymupdf() -> bool:
    """Guarantee PyMuPDF is importable, self-healing if it isn't.

    Called at startup so the Library's PDF viewer/ingest is reliably available
    on every launch regardless of how the app was started. If `import fitz`
    fails we make one best-effort `pip install pymupdf` into the *current*
    interpreter's environment, then re-import. Returns True if fitz ends up
    importable, False otherwise (logged, never raises — a missing optional dep
    must not abort startup).

    NOTE: PyMuPDF is AGPL-3.0. Auto-installing it here is a deliberate, opted-in
    choice for this self-hosted deployment; see requirements-optional.txt.
    """
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        pass
    logger.warning("PyMuPDF (fitz) not found — attempting one-time install for the PDF viewer…")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "PyMuPDF"],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except Exception as exc:
        logger.warning("PyMuPDF auto-install failed (%s): PDF viewer will be unavailable. "
                       "Install manually: pip install PyMuPDF", exc)
        return False
    try:
        import importlib
        import fitz  # noqa: F401
        importlib.invalidate_caches()
        logger.info("PyMuPDF installed and ready (%s)", getattr(fitz, "__doc__", "ok").splitlines()[0] if getattr(fitz, "__doc__", None) else "ok")
        return True
    except ImportError as exc:
        logger.warning("PyMuPDF still not importable after install (%s)", exc)
        return False
