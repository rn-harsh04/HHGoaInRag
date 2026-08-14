"""Unit tests for Reciprocal Rank Fusion (RRF)."""

from app.retrieval.fusion import rrf_fuse


def test_rrf_fuse_ranking():
    dense_hits = [
        ("chunk_A", 0.95),
        ("chunk_B", 0.88),
        ("chunk_C", 0.72),
    ]
    bm25_hits = [
        ("chunk_B", 12.5),
        ("chunk_A", 9.1),
        ("chunk_D", 8.0),
    ]

    fused = rrf_fuse([dense_hits, bm25_hits], k=60)

    # Chunk A and B appear in both lists, so they should rank higher than C or D
    assert len(fused) == 4
    top_ids = [hit.chunk_id for hit in fused]

    # chunk_A and chunk_B have the highest fused scores
    assert top_ids[0] in {"chunk_A", "chunk_B"}
    assert top_ids[1] in {"chunk_A", "chunk_B"}
    assert top_ids[2] in {"chunk_C", "chunk_D"}
    assert top_ids[3] in {"chunk_C", "chunk_D"}

    # Verify score formula: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    expected_score_a = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    score_a = next(h.score for h in fused if h.chunk_id == "chunk_A")
    assert abs(score_a - expected_score_a) < 1e-6


def test_rrf_fuse_empty_lists():
    fused = rrf_fuse([], k=60)
    assert fused == []

    fused_empty = rrf_fuse([[], []], k=60)
    assert fused_empty == []
