"""Unit tests for the 4 chunking strategies."""

import numpy as np
from indexing.chunking.strategies import (
    ParentRecord,
    fixed_overlap_chunks,
    semantic_chunks,
    metadata_chunks,
    child_chunks,
    all_chunks_for_parent,
)


def sample_parent() -> ParentRecord:
    return ParentRecord(
        parent_id="hi:1001:0",
        passage_id="hi:1001:0",
        query_id=1001,
        query_type="description",
        query_cluster=1,
        language_source="hi",
        text=(
            "Photosynthesis is the biological process by which green plants and certain other organisms "
            "transform light energy into chemical energy. During photosynthesis in green plants, light energy "
            "is captured and used to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds. "
            "It is crucial for life on Earth as it provides oxygen and forms the base of the food chain. "
            "Chloroplasts contain chlorophyll which absorbs sunlight and drives the entire metabolic cascade."
        ),
        alt_lang_text="प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा पौधे भोजन बनाते हैं।",
    )


def test_fixed_overlap_chunking():
    parent = sample_parent()
    chunks = fixed_overlap_chunks(parent, chunk_size=200, overlap=50)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.strategy == "fixed"
        assert chunk.parent_id == parent.parent_id
        assert chunk.query_id == parent.query_id
        assert len(chunk.text) <= 220
        assert chunk.char_start is not None
        assert chunk.char_end is not None


def test_semantic_chunking():
    parent = sample_parent()

    def mock_embed_fn(sentences: list[str]) -> np.ndarray:
        # Return deterministic dummy embeddings with unit norm
        np.random.seed(42)
        dim = 16
        vecs = np.random.randn(len(sentences), dim)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    chunks = semantic_chunks(parent, mock_embed_fn, similarity_threshold=0.5, max_chars=300)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.strategy == "semantic"
        assert chunk.parent_id == parent.parent_id
        assert len(chunk.text) > 0


def test_metadata_chunking():
    parent = sample_parent()
    chunks = metadata_chunks(parent)

    assert len(chunks) == 1
    assert chunks[0].strategy == "metadata"
    assert chunks[0].parent_id == parent.parent_id
    assert chunks[0].language_source == "hi"
    assert chunks[0].query_cluster == 1


def test_child_chunking():
    parent = sample_parent()
    chunks = child_chunks(parent, child_size=150, overlap=40)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.strategy == "child"
        assert chunk.parent_id == parent.parent_id
        assert len(chunk.text) <= 160


def test_all_chunks_generator():
    parent = sample_parent()

    def mock_embed_fn(sentences: list[str]) -> np.ndarray:
        dim = 8
        vecs = np.ones((len(sentences), dim))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    all_chunks = list(all_chunks_for_parent(parent, embed_fn=mock_embed_fn))
    strategies = {c.strategy for c in all_chunks}
    assert "fixed" in strategies
    assert "semantic" in strategies
    assert "metadata" in strategies
    assert "child" in strategies
