"""Agentic RAG Chat — FastAPI SSE streaming proxy to any OpenAI-compatible LLM."""
import json
import time
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

import config
import guardrails
import sessions
import tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

# Rate limit store (IP-based, in-memory)
_ip_hits: dict[str, list[float]] = {}

def check_ip_rate(ip: str) -> bool:
    now = time.time()
    hits = _ip_hits.setdefault(ip, [])
    hits[:] = [t for t in hits if now - t < 60]
    if len(hits) >= config.RATE_LIMIT_PER_MIN:
        return False
    hits.append(now)
    return True

# Auth dependency
async def verify_api_key(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = auth[7:]
    if token not in config.CLIENT_API_KEYS:
        raise HTTPException(403, "Invalid API key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agentic RAG Chat starting on port %s", config.PORT)
    yield
    logger.info("Agentic RAG Chat shutting down")

app = FastAPI(title="Agentic RAG Chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=config.MAX_MESSAGE_LENGTH)
    session_id: str | None = None

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "model": config.LLM_MODEL, "timestamp": time.time()}

@app.get("/api/v1/session/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_session(session_id: str):
    info = await sessions.session_info(session_id)
    if info is None:
        raise HTTPException(404, "Session not found")
    return info

@app.delete("/api/v1/session/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_session(session_id: str):
    await sessions.delete_session(session_id)
    return {"status": "deleted"}

@app.post("/api/v1/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile, source: str = Form(default=None)):
    """Upload and index a document for RAG retrieval."""
    import os
    import ingest_utils

    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ingest_utils.SUPPORTED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {ext}. Supported: pdf, txt, md, docx, csv")

    content = await file.read()
    if len(content) > ingest_utils.MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 50MB)")
    if not content:
        raise HTTPException(400, "Empty file")

    try:
        text = ingest_utils.extract_text(filename, content)
    except Exception as e:
        logger.error(f"Text extraction failed for {filename}: {e}")
        raise HTTPException(422, f"Could not extract text: {e}")

    if not text.strip():
        raise HTTPException(422, "No text content could be extracted from the file")

    try:
        src = source or filename
        num_chunks = ingest_utils.embed_and_store(text, src, ext.lstrip("."))
    except Exception as e:
        logger.error(f"Indexing failed for {filename}: {e}")
        raise HTTPException(500, f"Indexing failed: {e}")

    logger.info(f"Uploaded {filename}: {num_chunks} chunks indexed")
    return {
        "status": "ok",
        "filename": filename,
        "chunks": num_chunks,
        "message": "Document indexed successfully",
    }

@app.get("/api/v1/files", dependencies=[Depends(verify_api_key)])
async def list_files():
    """List all indexed documents in the RAG knowledge base."""
    import rag
    conn = rag._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT metadata->>'source' as source,
                       metadata->>'file_type' as file_type,
                       COUNT(*) as chunks,
                       MIN(metadata->>'upload_time') as uploaded
                FROM documents
                GROUP BY 1, 2
                ORDER BY uploaded DESC NULLS LAST, source
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        "files": [
            {"source": r[0], "file_type": r[1] or "", "chunks": r[2], "uploaded": r[3]}
            for r in rows
        ],
        "total": len(rows),
    }

@app.post("/api/v1/chat", dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest, request: Request):
    # Rate limit by IP
    client_ip = request.client.host if request.client else "unknown"
    if not check_ip_rate(client_ip):
        raise HTTPException(429, "Rate limit exceeded (20/min)")

    # Input guardrail
    blocked = guardrails.check_input(req.message)
    if blocked:
        async def blocked_stream():
            chunk = {
                "choices": [{"delta": {"content": blocked}, "index": 0, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    # Session
    sid = req.session_id
    if not sid:
        sid = await sessions.create_session()

    info = await sessions.get_session(sid)
    if info is None:
        sid = await sessions.create_session()

    # Session rate limit
    if not await sessions.check_rate_limit_session(sid, config.RATE_LIMIT_PER_HOUR_SESSION):
        raise HTTPException(429, "Session rate limit exceeded (100/hour)")

    # Get history and append user message
    history = await sessions.get_history(sid)
    history.append({"role": "user", "content": req.message})

    # Load system prompt and trim history
    system_prompt = config.load_system_prompt()

    # RAG: inject relevant documentation context
    try:
        import rag
        rag_context = rag.build_context(req.message)
        if rag_context:
            system_prompt = system_prompt + rag_context
            logger.info("RAG context injected (%d chars)", len(rag_context))
    except Exception as e:
        logger.warning("RAG unavailable: %s", e)

    history = await tokens.trim_history(system_prompt, history)

    # Build messages
    messages = [{"role": "system", "content": system_prompt}] + history

    # Save user message
    await sessions.append_message(sid, "user", req.message)

    # Stream from LLM backend
    async def stream():
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                async with client.stream(
                    "POST",
                    f"{config.LLM_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                    json={
                        "model": config.LLM_MODEL,
                        "messages": messages,
                        "stream": True,
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        logger.error(f"LLM error {resp.status_code}: {error_body[:500]}")
                        err = {"choices": [{"delta": {"content": "Error connecting to AI model."}, "index": 0, "finish_reason": None}]}
                        yield f"data: {json.dumps(err)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        if line == "data: [DONE]":
                            yield "data: [DONE]\n\n"
                            break

                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                clean = guardrails.sanitise_chunk(delta)
                                full_text += clean
                                if clean:
                                    data["choices"][0]["delta"]["content"] = clean
                                    data["session_id"] = sid
                                    yield f"data: {json.dumps(data)}\n\n"
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Stream error: {e}")
            err = {"choices": [{"delta": {"content": "Connection error. Please try again."}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

        # Output guardrail on full text
        final = guardrails.check_output_final(full_text)
        if final != full_text and final == guardrails.SAFE_RESPONSE:
            logger.warning("Output guardrail replaced entire response")

        # Save assistant response
        await sessions.append_message(sid, "assistant", final)
        await sessions.set_history(sid, history + [{"role": "assistant", "content": final}])

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "X-Session-ID": sid,
        "Cache-Control": "no-cache",
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
