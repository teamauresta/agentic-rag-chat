"""Redis session management with rate limiting."""
import json
import time
import uuid
import redis.asyncio as redis
from config import REDIS_URL, SESSION_TTL, MAX_MESSAGES_PER_SESSION

pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)

def get_redis():
    return redis.Redis(connection_pool=pool)

def _key(session_id: str) -> str:
    return f"agent:session:{session_id}"

async def create_session() -> str:
    sid = uuid.uuid4().hex[:16]
    r = get_redis()
    data = {"created": time.time(), "messages": []}
    await r.set(_key(sid), json.dumps(data), ex=SESSION_TTL)
    return sid

async def get_session(session_id: str) -> dict | None:
    r = get_redis()
    raw = await r.get(_key(session_id))
    if raw is None:
        return None
    return json.loads(raw)

async def get_history(session_id: str) -> list[dict]:
    data = await get_session(session_id)
    if data is None:
        return []
    return data.get("messages", [])

async def append_message(session_id: str, role: str, content: str):
    r = get_redis()
    key = _key(session_id)
    data = await get_session(session_id)
    if data is None:
        return
    msgs = data.get("messages", [])
    msgs.append({"role": role, "content": content})
    if len(msgs) > MAX_MESSAGES_PER_SESSION:
        msgs = msgs[-MAX_MESSAGES_PER_SESSION:]
    data["messages"] = msgs
    await r.set(key, json.dumps(data), ex=SESSION_TTL)

async def set_history(session_id: str, messages: list[dict]):
    """Replace history (after summarisation)."""
    r = get_redis()
    key = _key(session_id)
    data = await get_session(session_id)
    if data is None:
        return
    data["messages"] = messages
    await r.set(key, json.dumps(data), ex=SESSION_TTL)

async def delete_session(session_id: str):
    r = get_redis()
    await r.delete(_key(session_id))

async def session_info(session_id: str) -> dict | None:
    data = await get_session(session_id)
    if data is None:
        return None
    return {
        "session_id": session_id,
        "message_count": len(data.get("messages", [])),
        "created": data.get("created"),
    }

# Rate limiting
async def check_rate_limit_session(session_id: str, max_per_hour: int) -> bool:
    """Returns True if allowed."""
    r = get_redis()
    key = f"agent:rate:session:{session_id}"
    count = await r.incr(key)
    if count == 1:
        await r.expire(key, 3600)
    return count <= max_per_hour
