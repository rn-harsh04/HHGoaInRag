"""Benchmark trigger endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException

from app.api.deps import OrchestratorDep, SettingsDep
from app.schemas.query import BenchmarkReport

router = APIRouter(prefix="/v1", tags=["benchmark"])


@router.post("/benchmark/run", response_model=BenchmarkReport)
async def run_benchmark(
    settings: SettingsDep,
    orchestrator: OrchestratorDep,
    x_benchmark_key: str | None = Header(default=None),
) -> BenchmarkReport:
    if settings.benchmark_api_key and x_benchmark_key != settings.benchmark_api_key:
        raise HTTPException(status_code=401, detail="Invalid benchmark key")

    from benchmarks.run_benchmark import run_benchmark as execute

    queries_path = Path(__file__).resolve().parents[3] / "benchmarks" / "queries.json"
    if not queries_path.exists():
        raise HTTPException(status_code=404, detail="benchmarks/queries.json not found")

    report = execute(
        retriever=orchestrator.retriever if hasattr(orchestrator, "retriever") else None,
        queries_path=queries_path,
        runs=1,
        gate_ms=settings.retrieval_budget_ms,
        skip_llm=True,
    )
    return BenchmarkReport(**report)
