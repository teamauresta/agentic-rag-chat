#!/usr/bin/env python3
"""Document ingestion for the RAG pipeline.

Usage:
    python3 ingest.py --path /path/to/docs --source "Company FAQ"
    python3 ingest.py --path document.pdf --source "Product Manual"
    python3 ingest.py --path notes.txt --source "Internal Notes"
"""
import argparse
import os
import sys
import logging
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into chunks by approximate token count (words as proxy)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def read_file(path: Path) -> str:
    """Read a file's text content. Supports .txt, .md, .pdf."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            logger.warning("PyMuPDF not installed, trying pdfplumber...")
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                logger.error("No PDF library available. Install: pip install PyMuPDF")
                return ""
    else:
        return path.read_text(errors="replace")


def get_files(path: Path) -> list[Path]:
    """Get all ingestible files from a path."""
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf", ".rst", ".html"}
    return sorted(f for f in path.rglob("*") if f.suffix.lower() in exts and f.is_file())


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base")
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    parser.add_argument("--source", default="unknown", help="Source label (e.g. 'Product Manual')")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path not found: {path}")
        sys.exit(1)

    files = get_files(path)
    if not files:
        logger.error("No ingestible files found")
        sys.exit(1)

    logger.info(f"Found {len(files)} file(s) to ingest")

    # Load embedding model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Connect to DB
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("RAG_DB_HOST", "localhost"),
        port=int(os.getenv("RAG_DB_PORT", "5432")),
        user=os.getenv("RAG_DB_USER", "sotastack"),
        password=os.getenv("RAG_DB_PASS", "sotastack"),
        dbname=os.getenv("RAG_DB_NAME", "sotastack_agent"),
    )

    total_chunks = 0
    for f in files:
        logger.info(f"Processing: {f.name}")
        text = read_file(f)
        if not text.strip():
            logger.warning(f"  Skipping empty file: {f.name}")
            continue

        chunks = chunk_text(text, args.chunk_size, args.overlap)
        logger.info(f"  {len(chunks)} chunks")

        # Batch embed
        embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

        with conn.cursor() as cur:
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                metadata = {"source": args.source, "file": f.name, "chunk": i}
                cur.execute(
                    "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s::vector)",
                    (chunk, psycopg2.extras.Json(metadata), emb.tolist()),
                )
        conn.commit()
        total_chunks += len(chunks)

    conn.close()
    logger.info(f"Done! Ingested {total_chunks} chunks from {len(files)} files.")


if __name__ == "__main__":
    import psycopg2.extras
    main()
