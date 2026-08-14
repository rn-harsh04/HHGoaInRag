"""Download and normalize MSMARCO-XI subset."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset

from indexing.chunking.strategies import ParentRecord


def load_msmarco_subset(
    languages: list[str],
    queries_per_lang: int,
    seed: int = 42,
) -> list[ParentRecord]:
    rng = random.Random(seed)
    parents: list[ParentRecord] = []

    for lang in languages:
        ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split="validation")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = 0

        for idx in indices:
            if selected >= queries_per_lang:
                break
            row = ds[idx]
            passages = row.get("passages") or {}
            english = passages.get("English_passages") or []
            translated = passages.get("Translated_passages") or []
            query_id = int(row.get("query_id") or 0)
            query_type = str(row.get("query_type") or "unknown")

            added_for_query = False
            for p_idx, eng_text in enumerate(english):
                if not eng_text or not str(eng_text).strip():
                    continue
                alt = translated[p_idx] if p_idx < len(translated) else ""
                passage_id = f"{lang}:{query_id}:{p_idx}"
                parents.append(
                    ParentRecord(
                        parent_id=passage_id,
                        passage_id=passage_id,
                        query_id=query_id,
                        query_type=query_type,
                        query_cluster=-1,
                        language_source=lang,
                        text=str(eng_text).strip(),
                        alt_lang_text=str(alt).strip() if alt else "",
                    )
                )
                added_for_query = True

            if added_for_query:
                selected += 1

    return parents


def assign_query_clusters(parents: list[ParentRecord], k: int = 50) -> None:
    """Simple hash-based cluster assignment (fast; k-means optional upgrade)."""
    for p in parents:
        p.query_cluster = p.query_id % k


def save_parents(parents: list[ParentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_parents(path: Path) -> list[ParentRecord]:
    parents: list[ParentRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            parents.append(ParentRecord(**data))
    return parents
