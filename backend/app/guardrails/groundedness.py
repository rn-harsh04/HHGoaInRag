"""Post-generation groundedness checks."""

from __future__ import annotations

import re


def token_overlap_score(answer: str, contexts: list[str]) -> float:
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for ctx in contexts:
        context_tokens.update(re.findall(r"[a-z0-9]+", ctx.lower()))
    if not context_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def validate_chunk_ids(used_ids: list[str], retrieved_ids: set[str]) -> bool:
    if not used_ids:
        return False
    return any(cid in retrieved_ids for cid in used_ids)


