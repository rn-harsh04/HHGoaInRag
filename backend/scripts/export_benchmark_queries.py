"""Export benchmark queries from MSMARCO-XI validation Eng_Query fields."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def export_queries(languages: list[str], count: int, seed: int, output: Path) -> None:
    rng = random.Random(seed)
    queries: list[str] = []

    for lang in languages:
        ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split="validation")
        for row in ds:
            q = (row.get("Eng_Query") or "").strip()
            if q and len(q) > 10:
                queries.append(q)

    rng.shuffle(queries)
    selected = queries[:count]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(selected)} queries to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="hi,ta,bn")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./backend/benchmarks/queries.json")
    args = parser.parse_args()
    export_queries(
        [x.strip() for x in args.languages.split(",")],
        args.count,
        args.seed,
        Path(args.output),
    )
