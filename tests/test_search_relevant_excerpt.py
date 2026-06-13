"""Tests for the RAG layer over the web scraper: services/search/core.py
_relevant_excerpt picks the *most query-relevant* slice of a scraped page
instead of its leading N chars, and degrades gracefully without embeddings."""

import numpy as np

from services.search.core import (
    CONTENT_EXCERPT_BUDGET,
    _chunk_text_for_ranking,
    _relevant_excerpt,
)


class _NeedleEmbedClient:
    """Toy embedder: a text's vector is dominated by how many NEEDLEs it has,
    so a query containing NEEDLE scores highest against the chunk that has it."""

    def encode(self, texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            vecs.append([0.001, float(t.count("NEEDLE"))])
        arr = np.array(vecs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return arr / norms


class TestChunker:
    def test_short_text_single_chunk(self):
        assert _chunk_text_for_ranking("hello") == ["hello"]

    def test_empty_text_no_chunks(self):
        assert _chunk_text_for_ranking("   ") == []

    def test_windows_overlap_and_cover(self):
        text = "abcdefghij" * 200  # 2000 chars
        chunks = _chunk_text_for_ranking(text, chunk_size=700, overlap=100)
        assert len(chunks) >= 3
        # Overlap: end of one chunk reappears at the start of the next.
        assert chunks[0][-100:] == chunks[1][:100]


class TestRelevantExcerpt:
    def test_short_page_returned_untrimmed(self):
        text = "small page"
        out, trimmed = _relevant_excerpt("q", text, budget=CONTENT_EXCERPT_BUDGET)
        assert out == text and trimmed is False

    def test_picks_relevant_section_beyond_leading_slice(self):
        # The answer sits well past the first `budget` chars, so a leading slice
        # would miss it entirely. Semantic ranking must surface it.
        filler = "x" * 4000
        needle_block = " NEEDLE NEEDLE the-answer-is-here NEEDLE "
        text = filler + needle_block + "y" * 1000

        out, trimmed = _relevant_excerpt(
            "NEEDLE", text, budget=3000, embed_client=_NeedleEmbedClient()
        )
        assert trimmed is True
        assert "the-answer-is-here" in out
        assert len(out) <= 3000 + 200  # roughly within budget

    def test_falls_back_to_leading_slice_without_embeddings(self):
        text = "A" * 5000
        out, trimmed = _relevant_excerpt("q", text, budget=3000, embed_client=None)
        assert trimmed is True
        assert out == "A" * 3000

    def test_embed_failure_falls_back(self):
        class _Boom:
            def encode(self, *a, **k):
                raise RuntimeError("embeddings down")

        text = "B" * 5000
        out, trimmed = _relevant_excerpt("q", text, budget=3000, embed_client=_Boom())
        assert trimmed is True
        assert out == "B" * 3000
