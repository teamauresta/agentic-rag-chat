"""Three-layer guardrails: input filtering, streaming sanitisation, output validation."""
import re
import logging

logger = logging.getLogger("agent.guardrails")

SAFE_RESPONSE = "I'm happy to help. What would you like to know?"

# ── INPUT GUARDRAILS ──
INPUT_BLOCKED = [
    # Prompt extraction
    "system prompt", "your instructions", "your rules", "your configuration",
    "internal prompt", "original prompt", "initial prompt", "hidden prompt",
    "system message", "your prompt",
    # Injection
    "ignore previous", "ignore above", "disregard previous", "disregard",
    "jailbreak", "override", "developer mode", "dan mode", "debug mode",
    "root access", "admin access", "sudo",
]

def check_input(text: str) -> str | None:
    """Return blocked message or None if clean."""
    lower = text.lower()
    for pattern in INPUT_BLOCKED:
        if pattern in lower:
            logger.warning(f"INPUT GUARDRAIL triggered: '{pattern}' in message")
            return SAFE_RESPONSE
    return None

# ── OUTPUT GUARDRAILS ──
CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+')

LEAK_PATTERNS = [
    "system prompt", "my instructions", "my prompt", "i am configured",
    "i was instructed", "## security", "## communication style",
    "critical language rule", "absolute rules", "non-negotiable",
]

def sanitise_chunk(text: str) -> str:
    """Strip CJK characters from a streaming chunk (useful for multilingual models)."""
    return CJK_RE.sub('', text)

def check_output_final(text: str) -> str:
    """Check completed response for prompt leaks. Returns sanitised text."""
    clean = CJK_RE.sub('', text)
    lower = clean.lower()
    for pattern in LEAK_PATTERNS:
        if pattern in lower:
            logger.warning(f"OUTPUT GUARDRAIL triggered: '{pattern}' in response")
            return SAFE_RESPONSE
    return clean
