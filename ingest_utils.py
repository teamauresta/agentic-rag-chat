"""Shared ingestion utilities — chunking, embedding, storing."""
import os
import io
import csv
import logging
from datetime import datetime, timezone

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger = logging.getLogger("agent.ingest_utils")

SUPPORTED_TYPES = {".pdf", ".txt", ".md", ".docx", ".csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


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


def extract_text(filename: str, content: bytes) -> str:
    """Extract text from file bytes based on extension."""
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".txt", ".md"):
        return content.decode("utf-8", errors="replace")

    elif ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                raise RuntimeError("No PDF library available")

    elif ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".csv":
        text = content.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return ""
        headers = rows[0]
        parts = []
        for row in rows[1:]:
            pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v.strip()]
            parts.append("; ".join(pairs))
        return "\n".join(parts)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def embed_and_store(text: str, source: str, file_type: str) -> int:
    """Chunk text, embed on CPU, store in pgvector. Returns chunk count."""
    import psycopg2
    import psycopg2.extras
    import rag

    chunks = chunk_text(text)
    if not chunks:
        return 0

    model = rag._get_model()
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    conn = rag._get_conn()
    try:
        with conn.cursor() as cur:
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "source": source,
                    "file_type": file_type,
                    "upload_time": datetime.now(timezone.utc).isoformat(),
                    "chunk": i,
                }
                cur.execute(
                    "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s::vector)",
                    (chunk, psycopg2.extras.Json(metadata), emb.tolist()),
                )
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Stored {len(chunks)} chunks from {source}")
    return len(chunks)
