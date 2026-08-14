"""Pre-retrieval safety checks."""

from __future__ import annotations

import re

BLOCKED_PATTERNS = [
    r"\b(bomb|kill|terror|suicide)\b",
    r"\b(porn|nude|sexual)\b",
    r"\b(hack\s+credit\s+card)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


def is_unsafe(text: str) -> tuple[bool, str | None]:
    normalized = text.strip().lower()
    if len(normalized) < 2:
        return True, "Query too short"
    for pattern in _COMPILED:
        if pattern.search(normalized):
            return True, "Unsafe or inappropriate content detected"
    return False, None


def is_off_topic(max_score: float, threshold: float) -> bool:
    return max_score < threshold
