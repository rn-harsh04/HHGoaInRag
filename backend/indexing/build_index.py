"""Build Chroma index, export FAISS, and serialize BM25."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import chromadb
import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Allow running as module from backend/
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.embeddings import EmbeddingService
from app.services.tokenizer import tokenize
from indexing.chunking.strategies import ChunkRecord, all_chunks_for_parent
from indexing.normalize import assign_query_clusters, load_msmarco_subset, save_parents


def build_index(
    languages: list[str],
    queries_per_lang: int,
    chroma_dir: Path,
    faiss_path: Path,
    id_map_path: Path,
    chunk_parquet: Path,
    parents_path: Path,
    bm25_dir: Path,
    embedding_model: str,
) -> None:
    print(f"Loading MSMARCO-XI subset: {languages}, {queries_per_lang}/lang")
    parents = load_msmarco_subset(languages, queries_per_lang)
    assign_query_clusters(parents)
    save_parents(parents, parents_path)
    print(f"Parents: {len(parents)}")

    embedder = EmbeddingService(embedding_model)

    def embed_sentences(texts: list[str]) -> np.ndarray:
        return embedder.embed_passages(texts)

    chunks: list[ChunkRecord] = []
    for parent in parents:
        chunks.extend(list(all_chunks_for_parent(parent, embed_fn=embed_sentences)))

    print(f"Total chunks: {len(chunks)}")
    if len(chunks) > 45000:
        raise RuntimeError(f"Chunk count {len(chunks)} exceeds 45K SLA cap")

    texts = [c.text for c in chunks]
    print("Embedding chunks...")
    vectors = embedder.embed_passages(texts)

    # Chroma persistence
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection("chunks")
    except Exception:
        pass
    collection = client.create_collection(name="chunks", metadata={"hnsw:space": "cosine"})

    batch = 500
    for i in range(0, len(chunks), batch):
        batch_chunks = chunks[i : i + batch]
        batch_vecs = vectors[i : i + batch].tolist()
        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_vecs,
            documents=[c.text for c in batch_chunks],
            metadatas=[
                {
                    "strategy": c.strategy,
                    "parent_id": c.parent_id,
                    "passage_id": c.passage_id,
                    "query_id": c.query_id,
                    "query_type": c.query_type,
                    "query_cluster": c.query_cluster,
                    "language_source": c.language_source,
                }
                for c in batch_chunks
            ],
        )
    print(f"Chroma collection stored at {chroma_dir}")

    # FAISS export
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 32
    index.add(vectors)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))

    id_map = [c.chunk_id for c in chunks]
    id_map_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")

    # Chunk metadata parquet
    chunk_parquet.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(c) for c in chunks])
    df.to_parquet(chunk_parquet, index=False)

    # BM25
    bm25_dir.mkdir(parents=True, exist_ok=True)
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with (bm25_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": id_map}, f)

    print(f"FAISS index: {faiss_path} ({len(id_map)} vectors, dim={dim})")
    print("Index build complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="hi,ta,bn")
    parser.add_argument("--queries-per-lang", type=int, default=500)
    parser.add_argument("--chroma-dir", default="./data/chroma")
    parser.add_argument("--faiss-path", default="./data/faiss/hnsw.index")
    parser.add_argument("--id-map-path", default="./data/faiss/id_map.json")
    parser.add_argument("--chunk-parquet", default="./data/processed/chunks.parquet")
    parser.add_argument("--parents-path", default="./data/processed/parents.jsonl")
    parser.add_argument("--bm25-dir", default="./data/bm25")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    build_index(
        languages=[x.strip() for x in args.languages.split(",")],
        queries_per_lang=args.queries_per_lang,
        chroma_dir=Path(args.chroma_dir),
        faiss_path=Path(args.faiss_path),
        id_map_path=Path(args.id_map_path),
        chunk_parquet=Path(args.chunk_parquet),
        parents_path=Path(args.parents_path),
        bm25_dir=Path(args.bm25_dir),
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
