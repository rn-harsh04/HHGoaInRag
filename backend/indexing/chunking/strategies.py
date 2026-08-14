"""Chunking strategy implementations."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class ChunkRecord:
    chunk_id: str
    strategy: str
    parent_id: str
    passage_id: str
    query_id: int
    query_type: str
    query_cluster: int
    language_source: str
    text: str
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class ParentRecord:
    parent_id: str
    passage_id: str
    query_id: int
    query_type: str
    query_cluster: int
    language_source: str
    text: str
    alt_lang_text: str = ""


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def fixed_overlap_chunks(
    parent: ParentRecord,
    chunk_size: int = 512,
    overlap: int = 128,
) -> list[ChunkRecord]:
    text = parent.text
    if not text:
        return []
    chunks: list[ChunkRecord] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + chunk_size // 2:
                end = space
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    strategy="fixed",
                    parent_id=parent.parent_id,
                    passage_id=parent.passage_id,
                    query_id=parent.query_id,
                    query_type=parent.query_type,
                    query_cluster=parent.query_cluster,
                    language_source=parent.language_source,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def semantic_chunks(
    parent: ParentRecord,
    embed_fn,
    similarity_threshold: float = 0.75,
    max_chars: int = 512,
) -> list[ChunkRecord]:
    sentences = _split_sentences(parent.text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return fixed_overlap_chunks(parent, chunk_size=max_chars, overlap=0)

    embeddings = embed_fn(sentences)
    merged: list[str] = []
    current = sentences[0]
    current_embs = [embeddings[0]]

    for i in range(1, len(sentences)):
        avg_current = np.mean(current_embs, axis=0)
        sim = float(np.dot(avg_current, embeddings[i]))
        candidate = f"{current} {sentences[i]}"
        if sim >= similarity_threshold and len(candidate) <= max_chars:
            current = candidate
            current_embs.append(embeddings[i])
        else:
            merged.append(current)
            current = sentences[i]
            current_embs = [embeddings[i]]
    merged.append(current)

    return [
        ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            strategy="semantic",
            parent_id=parent.parent_id,
            passage_id=parent.passage_id,
            query_id=parent.query_id,
            query_type=parent.query_type,
            query_cluster=parent.query_cluster,
            language_source=parent.language_source,
            text=m.strip(),
        )
        for m in merged
        if m.strip()
    ]


def metadata_chunks(parent: ParentRecord) -> list[ChunkRecord]:
    """Whole passage if short, else one fixed chunk tagged metadata strategy."""
    text = parent.text.strip()
    if not text:
        return []
    if len(text) <= 512:
        body = text
    else:
        body = text[:512].rsplit(" ", 1)[0]
    return [
        ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            strategy="metadata",
            parent_id=parent.parent_id,
            passage_id=parent.passage_id,
            query_id=parent.query_id,
            query_type=parent.query_type,
            query_cluster=parent.query_cluster,
            language_source=parent.language_source,
            text=body,
        )
    ]


def child_chunks(
    parent: ParentRecord,
    child_size: int = 256,
    overlap: int = 64,
) -> list[ChunkRecord]:
    text = parent.text
    if not text:
        return []
    chunks: list[ChunkRecord] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + child_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    strategy="child",
                    parent_id=parent.parent_id,
                    passage_id=parent.passage_id,
                    query_id=parent.query_id,
                    query_type=parent.query_type,
                    query_cluster=parent.query_cluster,
                    language_source=parent.language_source,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def all_chunks_for_parent(
    parent: ParentRecord,
    embed_fn=None,
) -> Iterator[ChunkRecord]:
    yield from fixed_overlap_chunks(parent)
    if embed_fn is not None:
        yield from semantic_chunks(parent, embed_fn)
    yield from metadata_chunks(parent)
    yield from child_chunks(parent)
