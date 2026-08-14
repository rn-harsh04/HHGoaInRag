"""Reciprocal rank fusion for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredHit:
    chunk_id: str
    score: float
    source: str


def rrf_fuse(lists: list[list[tuple[str, float]]], *, k: int = 60) -> list[ScoredHit]:
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}

    for lst in lists:
        for rank, (chunk_id, _raw) in enumerate(lst, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(chunk_id, set()).add("fused")

    fused = [
        ScoredHit(chunk_id=cid, score=score, source=",".join(sorted(sources.get(cid, {"unknown"}))))
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused
