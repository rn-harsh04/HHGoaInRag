"""Unit tests for HybridRetriever (FAISS + BM25 + RRF)."""

import pytest
from app.config import get_settings
from app.retrieval.retriever import HybridRetriever
from app.services.embeddings import EmbeddingService
from scripts.bootstrap_sample_data import bootstrap_sample_index


@pytest.fixture(scope="session", autouse=True)
def setup_index():
    bootstrap_sample_index()


@pytest.mark.asyncio
async def test_hybrid_retrieval():
    settings = get_settings()
    embedder = EmbeddingService.get_instance(settings)
    retriever = HybridRetriever.load(settings, embedder)

    payload, stats = await retriever.retrieve("What is the capital of India?")

    assert payload is not None
    assert len(payload.chunks) > 0
    assert payload.max_score > 0
    assert payload.latency_ms <= 200  # Strict SLA verification

    # Check top hit relevance
    top_chunk_text = payload.chunks[0].text.lower()
    assert "delhi" in top_chunk_text or "india" in top_chunk_text

    # Verify parent context resolution
    assert len(payload.parents_used) > 0
    assert any("delhi" in p.text.lower() for p in payload.parents_used)


@pytest.mark.asyncio
async def test_retrieval_strategies_represented():
    settings = get_settings()
    embedder = EmbeddingService.get_instance(settings)
    retriever = HybridRetriever.load(settings, embedder)

    payload, stats = await retriever.retrieve("How does photosynthesis work?")
    strategies = {c.strategy for c in payload.chunks}

    # Verify chunks have valid strategy tags
    assert len(strategies) > 0
    assert any(s in {"fixed", "semantic", "metadata", "child"} for s in strategies)
