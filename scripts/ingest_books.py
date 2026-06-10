#!/usr/bin/env python3
"""
Ingest coding books into Qdrant for Claude Code RAG.

Prerequisites:
  docker run -d --name qdrant -p 6333:6333 -v ~/.qdrant:/qdrant/storage qdrant/qdrant
  pip install qdrant-client fastembed pypdf

Usage:
  python scripts/ingest_books.py [--books-dir PATH] [--collection COLLECTION]

Defaults to the books directory at:
  /media/dari/Above-Average/RAG Data/RAG Data/
and collection name: coding_books

Run again after adding new books — already-ingested IDs are skipped.
"""

import argparse
import hashlib
import sys
from pathlib import Path

BOOKS_DIR_DEFAULT = "/media/dari/Above-Average/RAG Data/RAG Data"
COLLECTION_DEFAULT = "coding_books"
QDRANT_URL_DEFAULT = "http://localhost:6333"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return chunks


def ingest_pdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        print(f"  [skip] {path.name}: {e}", file=sys.stderr)
        return []


def ingest_text(path: Path) -> list[str]:
    try:
        return [path.read_text(errors="ignore")]
    except Exception as e:
        print(f"  [skip] {path.name}: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--books-dir", default=BOOKS_DIR_DEFAULT)
    parser.add_argument("--collection", default=COLLECTION_DEFAULT)
    parser.add_argument("--qdrant-url", default=QDRANT_URL_DEFAULT)
    args = parser.parse_args()

    books_dir = Path(args.books_dir)
    if not books_dir.exists():
        sys.exit(f"Books directory not found: {books_dir}")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from fastembed import TextEmbedding
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nRun: pip install qdrant-client fastembed pypdf")

    print(f"Connecting to Qdrant at {args.qdrant_url}...")
    client = QdrantClient(url=args.qdrant_url)

    print(f"Loading embedding model {EMBED_MODEL}...")
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    vector_size = 384  # all-MiniLM-L6-v2

    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        client.create_collection(
            args.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection '{args.collection}'")

    extensions = {".pdf", ".txt", ".md", ".epub"}
    book_files = [p for p in books_dir.rglob("*") if p.suffix.lower() in extensions]
    print(f"Found {len(book_files)} files in {books_dir}")

    total_chunks = 0
    for book_path in book_files:
        print(f"  {book_path.name}")
        raw_pages = ingest_pdf(book_path) if book_path.suffix.lower() == ".pdf" else ingest_text(book_path)
        full_text = "\n".join(raw_pages)
        if not full_text.strip():
            continue

        chunks = chunk_text(full_text)
        points = []
        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{book_path}:{idx}:{chunk[:64]}".encode()).hexdigest()
            # Use first 16 hex chars as a reproducible uint64 ID
            point_id = int(chunk_id[:16], 16)
            vectors = list(embedder.embed([chunk]))[0].tolist()
            points.append(PointStruct(
                id=point_id,
                vector=vectors,
                payload={"source": str(book_path), "chunk": idx, "text": chunk},
            ))

        if points:
            client.upsert(collection_name=args.collection, points=points)
            total_chunks += len(points)
            print(f"    → {len(points)} chunks upserted")

    print(f"\nDone. {total_chunks} total chunks in '{args.collection}'.")


if __name__ == "__main__":
    main()
