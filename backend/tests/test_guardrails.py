"""Unit tests for pre and post guardrails."""

import json
from pathlib import Path
from app.guardrails.safety import is_unsafe, is_off_topic
from app.guardrails.groundedness import token_overlap_score, validate_chunk_ids
from app.guardrails.refusal_logger import log_refusal


def test_pre_guardrail_safety():
    # Safe queries
    unsafe, reason = is_unsafe("What is the capital of India?")
    assert not unsafe
    assert reason is None

    unsafe, reason = is_unsafe("How do solar panels generate electricity?")
    assert not unsafe

    # Unsafe queries
    unsafe, reason = is_unsafe("How to make a bomb at home?")
    assert unsafe
    assert "Unsafe" in reason

    unsafe, reason = is_unsafe("where to find free porn videos")
    assert unsafe

    # Too short
    unsafe, reason = is_unsafe("a")
    assert unsafe
    assert "short" in reason.lower()


def test_off_topic_detection():
    # Below threshold is off topic
    assert is_off_topic(max_score=0.25, threshold=0.42) is True
    # Above threshold is on topic
    assert is_off_topic(max_score=0.75, threshold=0.42) is False


def test_groundedness_token_overlap():
    contexts = [
        "Photosynthesis allows plants to transform sunlight into glucose.",
        "Oxygen is released as a byproduct during the metabolic process."
    ]

    # Good overlap
    answer_good = "Photosynthesis transforms sunlight into glucose in plants."
    score_good = token_overlap_score(answer_good, contexts)
    assert score_good > 0.60

    # Hallucinated answer
    answer_hallucinated = "Quantum mechanics governs black hole event horizons."
    score_bad = token_overlap_score(answer_hallucinated, contexts)
    assert score_bad < 0.20


def test_validate_chunk_ids():
    retrieved = {"chunk_1", "chunk_2", "chunk_3"}

    assert validate_chunk_ids(["chunk_1", "chunk_2"], retrieved) is True
    assert validate_chunk_ids(["chunk_4"], retrieved) is False
    assert validate_chunk_ids([], retrieved) is False


def test_refusal_logger(tmp_path: Path):
    log_file = tmp_path / "refusals.jsonl"
    log_refusal(
        log_file,
        request_id="test-req-123",
        query="unsafe query text",
        reason="unsafe_input",
        stage="pre_guardrail",
        max_score=0.1,
    )

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["request_id"] == "test-req-123"
    assert data["reason"] == "unsafe_input"
