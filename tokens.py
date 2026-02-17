"""Token counting and history trimming with automatic summarisation."""
import logging
import httpx
import tiktoken
from config import LLM_URL, LLM_API_KEY, LLM_MODEL, MAX_TOKENS_CONTEXT

logger = logging.getLogger("agent.tokens")
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", "")) + 4  # role overhead
    return total

async def summarise_messages(messages: list[dict]) -> str:
    """Summarise a list of messages via the LLM backend."""
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = (
        "Summarise this conversation concisely, preserving key facts, "
        "decisions, and technical details. Keep it under 500 words:\n\n" + text
    )
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{LLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.3,
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

async def trim_history(system_prompt: str, messages: list[dict]) -> list[dict]:
    """Trim history to fit within token budget. Returns new message list."""
    sys_tokens = count_tokens(system_prompt) + 4
    budget = MAX_TOKENS_CONTEXT - sys_tokens

    if count_messages_tokens(messages) <= budget:
        return messages

    logger.info("History exceeds token budget, summarising...")
    keep = 6
    recent = messages[-keep:] if len(messages) > keep else messages
    old = messages[:-keep] if len(messages) > keep else []

    if not old:
        return recent

    try:
        summary = await summarise_messages(old)
        summary_msg = {"role": "system", "content": f"[Previous conversation summary]: {summary}"}
        trimmed = [summary_msg] + recent

        while count_messages_tokens(trimmed) > budget and len(trimmed) > 2:
            trimmed.pop(1)

        return trimmed
    except Exception as e:
        logger.error(f"Summarisation failed: {e}")
        return recent
