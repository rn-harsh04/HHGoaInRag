"""Refusal audit logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_refusal(
    path: Path,
    *,
    request_id: str,
    query: str,
    reason: str,
    stage: str,
    max_score: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "query": query,
        "reason": reason,
        "stage": stage,
        "max_score": max_score,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
