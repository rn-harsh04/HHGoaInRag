"""Unit and integration tests for the VoiceRAGOrchestrator pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.config import get_settings
from app.orchestrator.pipeline import VoiceRAGOrchestrator
from app.schemas.query import TextQueryRequest, ResponseStatus
from app.services.gemini_client import GeminiAnswerSchema
from app.retrieval.retriever import HybridRetriever
from app.services.embeddings import EmbeddingService
from scripts.bootstrap_sample_data import bootstrap_sample_index


@pytest.fixture(scope="session", autouse=True)
def setup_index():
    bootstrap_sample_index()


@pytest.mark.asyncio
async def test_orchestrator_text_query_success():
    settings = get_settings()
    embedder = EmbeddingService.get_instance(settings)
    retriever = HybridRetriever.load(settings, embedder)

    # Mock STT and Gemini
    mock_stt = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.generate = AsyncMock(
        return_value=(
            GeminiAnswerSchema(
                answer="New Delhi is the capital of India.",
                used_chunk_ids=[],  # Will be populated with retrieved ID
                confidence=0.95,
                refused=False,
            ),
            120.0,
        )
    )

    orchestrator = VoiceRAGOrchestrator(settings, retriever, mock_stt, mock_gemini)

    # First perform retrieval to get valid retrieved ID for mock
    retrieval_res, _ = await retriever.retrieve("capital of India")
    assert len(retrieval_res.chunks) > 0
    valid_id = retrieval_res.chunks[0].chunk_id
    mock_gemini.generate.return_value = (
        GeminiAnswerSchema(
            answer="New Delhi is the capital of India.",
            used_chunk_ids=[valid_id],
            confidence=0.95,
            refused=False,
        ),
        120.0,
    )

    req = TextQueryRequest(query="What is the capital of India?", language="en")
    resp = await orchestrator.run_text_query(req)

    assert resp.status == ResponseStatus.SUCCESS
    assert "Delhi" in resp.answer.text
    assert resp.groundedness.passed is True
    assert resp.timings_ms.retrieval <= 200


@pytest.mark.asyncio
async def test_orchestrator_unsafe_query_refusal():
    settings = get_settings()
    embedder = EmbeddingService.get_instance(settings)
    retriever = HybridRetriever.load(settings, embedder)

    mock_stt = MagicMock()
    mock_gemini = MagicMock()
    orchestrator = VoiceRAGOrchestrator(settings, retriever, mock_stt, mock_gemini)

    req = TextQueryRequest(query="How to make a bomb?", language="en")
    resp = await orchestrator.run_text_query(req)

    assert resp.status == ResponseStatus.REFUSAL_UNSAFE
    assert resp.answer.refused is True
    # LLM should not be called when safety guardrail triggers
    mock_gemini.generate.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_off_topic_refusal():
    settings = get_settings()
    embedder = EmbeddingService.get_instance(settings)
    retriever = HybridRetriever.load(settings, embedder)

    mock_stt = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.generate = AsyncMock(
        return_value=(
            GeminiAnswerSchema(
                answer="I don't have enough information.",
                used_chunk_ids=[],
                confidence=0.0,
                refused=True,
                refusal_reason="Insufficient context",
            ),
            50.0,
        )
    )

    # Set high retrieval threshold to guarantee off-topic trigger
    original_threshold = settings.retrieval_min_score
    settings.retrieval_min_score = 0.99
    try:
        orchestrator = VoiceRAGOrchestrator(settings, retriever, mock_stt, mock_gemini)
        req = TextQueryRequest(
            query="xyzjkfdskfsdjfkds completely random gibberish 9342423984", language="en"
        )
        resp = await orchestrator.run_text_query(req)

        assert resp.status == ResponseStatus.REFUSAL_INSUFFICIENT_INFO
        assert resp.answer.refused is True
        mock_gemini.generate.assert_not_called()
    finally:
        settings.retrieval_min_score = original_threshold

