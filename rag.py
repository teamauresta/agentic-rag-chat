"""RAG module — CPU-only embedding + pgvector retrieval."""
import os
import time
import logging
import numpy as np

logger = logging.getLogger("agent.rag")

RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.3"))

DB_HOST = os.getenv("RAG_DB_HOST", "localhost")
DB_PORT = int(os.getenv("RAG_DB_PORT", "5432"))
DB_USER = os.getenv("RAG_DB_USER", "postgres")
DB_PASS = os.getenv("RAG_DB_PASS", "postgres")
DB_NAME = os.getenv("RAG_DB_NAME", "agentic_rag")

_model = None
_pool = None


def _get_model():
    global _model
    if _model is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Loaded embedding model (CPU)")
    return _model


def embed(text: str) -> list[float]:
    """Embed a single text string, returns list of 384 floats."""
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def _get_conn():
    import psycopg2
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS,
        dbname=DB_NAME,
    )


def search(query: str, top_k: int = None, min_similarity: float = None) -> list[dict]:
    """Embed query and search pgvector for similar chunks."""
    if top_k is None:
        top_k = RAG_TOP_K
    if min_similarity is None:
        min_similarity = RAG_MIN_SIMILARITY

    t0 = time.time()
    vec = embed(query)
    t_embed = time.time() - t0

    t1 = time.time()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM documents
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vec, vec, min_similarity, vec, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    t_search = time.time() - t1
    logger.info(f"RAG: embed={t_embed*1000:.0f}ms search={t_search*1000:.0f}ms results={len(rows)}")

    return [
        {"content": r[0], "metadata": r[1], "similarity": float(r[2])}
        for r in rows
    ]


def build_context(query: str) -> str | None:
    """Search and format context for injection into system prompt. Returns None if no results."""
    if not RAG_ENABLED:
        return None
    try:
        results = search(query)
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return None

    if not results:
        return None

    parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"] or {}
        source = meta.get("source", "unknown")
        ftype = meta.get("file_type", "")
        label = f"{source} ({ftype})" if ftype else source
        parts.append(f"[{i}] (source: {label}, similarity: {r['similarity']:.2f})\n{r['content']}")

    return "\n\n## REFERENCE DATA\nThe following information is from your knowledge base. Use this data to inform your answers. If there is a conflict between your training data and the reference data below, prefer the reference data.\n\n" + "\n\n---\n\n".join(parts)
