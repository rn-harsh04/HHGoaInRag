"""Benchmark runner with SLA gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.logging.timing import compute_stats
from app.retrieval.retriever import HybridRetriever
from app.services.embeddings import EmbeddingService


async def _run_queries(retriever: HybridRetriever, queries: list[str], runs: int) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = {
        "embed_query_ms": [],
        "dense_search_ms": [],
        "bm25_search_ms": [],
        "fusion_ms": [],
        "parent_resolve_ms": [],
        "retrieval_total_ms": [],
    }

    for _ in range(runs):
        for q in queries:
            payload, stats = await retriever.retrieve(q)
            buckets["embed_query_ms"].append(stats.embed_query_ms)
            buckets["dense_search_ms"].append(stats.dense_search_ms)
            buckets["bm25_search_ms"].append(stats.bm25_search_ms)
            buckets["fusion_ms"].append(stats.fusion_ms)
            buckets["parent_resolve_ms"].append(stats.parent_resolve_ms)
            buckets["retrieval_total_ms"].append(payload.latency_ms)

    return buckets


def run_benchmark(
    *,
    retriever: HybridRetriever | None,
    queries_path: Path,
    runs: int = 3,
    gate_ms: int = 200,
    skip_llm: bool = True,
    output_path: Path | None = None,
) -> dict:
    settings = get_settings()
    if retriever is None:
        embedder = EmbeddingService.get_instance(settings)
        retriever = HybridRetriever.load(settings, embedder)
        embedder.warmup(settings.warmup_queries)

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list):
        raise ValueError("queries.json must be a list of strings")

    buckets = asyncio.run(_run_queries(retriever, queries, runs))

    report = {
        "sla_target_ms": gate_ms,
        "query_count": len(queries),
        "runs": runs,
        "skip_llm": skip_llm,
    }

    for stage, values in buckets.items():
        report[stage] = compute_stats(values)

    p100 = report["retrieval_total_ms"]["p100"]
    report["sla_pass"] = p100 <= gate_ms

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_chart(report, output_path.with_suffix(".png"))

    return report


def _write_chart(report: dict, chart_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        stages = [
            "embed_query_ms",
            "dense_search_ms",
            "bm25_search_ms",
            "fusion_ms",
            "parent_resolve_ms",
            "retrieval_total_ms",
        ]
        p50 = [report[s]["p50"] for s in stages]
        labels = [s.replace("_ms", "") for s in stages]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, p50, color="#4f46e5")
        ax.axhline(report["sla_target_ms"], color="red", linestyle="--", label="SLA")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Retrieval Benchmark (P50)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(chart_path)
        plt.close(fig)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(BACKEND_ROOT / "benchmarks" / "queries.json"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gate", type=int, default=200)
    parser.add_argument("--output", default="./reports/latency.json")
    args = parser.parse_args()

    report = run_benchmark(
        retriever=None,
        queries_path=Path(args.queries),
        runs=args.runs,
        gate_ms=args.gate,
        output_path=Path(args.output),
    )

    print(json.dumps(report, indent=2))
    if not report["sla_pass"]:
        print(f"SLA FAILED: retrieval P100={report['retrieval_total_ms']['p100']:.1f}ms > {args.gate}ms")
        sys.exit(1)
    print("SLA PASSED")


if __name__ == "__main__":
    main()
