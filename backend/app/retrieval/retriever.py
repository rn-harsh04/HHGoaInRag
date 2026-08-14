"""In-memory FAISS + BM25 hybrid retriever with strict latency budget."""

from __future__ import annotations

import asyncio
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from app.config import Settings
from app.logging.timing import StageTimer
from app.retrieval.fusion import ScoredHit, rrf_fuse
from app.schemas.query import ChunkHit, ParentContext, RetrievalPayload
from app.services.embeddings import EmbeddingService
from app.services.tokenizer import tokenize
from indexing.normalize import ParentRecord, load_parents


@dataclass
class RetrievalStats:
    embed_query_ms: float = 0.0
    dense_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    fusion_ms: float = 0.0
    parent_resolve_ms: float = 0.0


@dataclass
class HybridRetriever:
    settings: Settings
    embedder: EmbeddingService
    faiss_index: faiss.Index
    id_map: list[str]
    chunk_df: pd.DataFrame
    chunk_lookup: dict[str, dict]
    parents: dict[str, ParentRecord]
    bm25: object
    bm25_ids: list[str]
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=2))

    @classmethod
    def load(cls, settings: Settings, embedder: EmbeddingService) -> HybridRetriever:
        if not settings.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {settings.faiss_index_path}")

        index = faiss.read_index(str(settings.faiss_index_path))
        id_map = json.loads(settings.faiss_id_map_path.read_text(encoding="utf-8"))
        chunk_df = pd.read_parquet(settings.chunk_metadata_path)
        chunk_lookup = {row["chunk_id"]: row for row in chunk_df.to_dict(orient="records")}

        parents_list = load_parents(settings.parents_path)
        parents = {p.parent_id: p for p in parents_list}

        with (settings.bm25_index_dir / "bm25.pkl").open("rb") as f:
            bm25_data = pickle.load(f)

        return cls(
            settings=settings,
            embedder=embedder,
            faiss_index=index,
            id_map=id_map,
            chunk_df=chunk_df,
            chunk_lookup=chunk_lookup,
            parents=parents,
            bm25=bm25_data["bm25"],
            bm25_ids=bm25_data["chunk_ids"],
        )

    def _dense_search(self, q_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        with StageTimer("dense") as t:
            scores, indices = self.faiss_index.search(q_vec.reshape(1, -1), top_k)
        self._last_dense_ms = t.elapsed_ms
        hits: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            hits.append((self.id_map[idx], float(score)))
        return hits

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        with StageTimer("bm25") as t:
            tokens = tokenize(query)
            scores = self.bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
        self._last_bm25_ms = t.elapsed_ms
        return [(self.bm25_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _retrieve_sync(self, query: str) -> tuple[RetrievalPayload, RetrievalStats]:
        stats = RetrievalStats()
        budget = self.settings.retrieval_budget_ms

        with StageTimer("embed", budget_ms=budget) as embed_timer:
            q_vec = self.embedder.embed_query(query)
        stats.embed_query_ms = embed_timer.elapsed_ms

        dense_hits = self._dense_search(q_vec, self.settings.top_k_per_channel)
        stats.dense_search_ms = getattr(self, "_last_dense_ms", 0.0)

        bm25_hits = self._bm25_search(query, self.settings.top_k_per_channel)
        stats.bm25_search_ms = getattr(self, "_last_bm25_ms", 0.0)

        with StageTimer("fusion", budget_ms=budget) as fusion_timer:
            fused: list[ScoredHit] = rrf_fuse([dense_hits, bm25_hits], k=self.settings.rrf_k)
        stats.fusion_ms = fusion_timer.elapsed_ms

        max_dense = dense_hits[0][1] if dense_hits else 0.0

        with StageTimer("parents", budget_ms=budget) as parent_timer:
            chunk_hits: list[ChunkHit] = []
            seen_parents: set[str] = set()
            parents_used: list[ParentContext] = []

            for rank, hit in enumerate(fused[: self.settings.final_top_k], start=1):
                meta = self.chunk_lookup.get(hit.chunk_id)
                if not meta:
                    continue
                chunk_hits.append(
                    ChunkHit(
                        chunk_id=hit.chunk_id,
                        strategy=str(meta.get("strategy", "")),
                        text=str(meta.get("text", "")),
                        score=hit.score,
                        parent_id=str(meta.get("parent_id", "")),
                        passage_id=str(meta.get("passage_id", "")),
                        rank=rank,
                    )
                )
                pid = str(meta.get("parent_id", ""))
                if pid and pid not in seen_parents and len(parents_used) < self.settings.max_parents:
                    parent = self.parents.get(pid)
                    if parent:
                        seen_parents.add(pid)
                        parents_used.append(
                            ParentContext(
                                parent_id=parent.parent_id,
                                text=parent.text,
                                passage_id=parent.passage_id,
                                language_source=parent.language_source,
                            )
                        )
        stats.parent_resolve_ms = parent_timer.elapsed_ms

        total_ms = (
            stats.embed_query_ms
            + max(stats.dense_search_ms, stats.bm25_search_ms)
            + stats.fusion_ms
            + stats.parent_resolve_ms
        )

        return (
            RetrievalPayload(
                chunks=chunk_hits,
                parents_used=parents_used,
                max_score=max_dense,
                latency_ms=total_ms,
                timed_out=total_ms > budget,
            ),
            stats,
        )

    async def retrieve(self, query: str) -> tuple[RetrievalPayload, RetrievalStats]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._retrieve_sync, query)

    @property
    def chunk_count(self) -> int:
        return len(self.id_map)

    @property
    def parent_count(self) -> int:
        return len(self.parents)
