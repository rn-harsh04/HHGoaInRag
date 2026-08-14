"""Bootstrap sample dataset and build initial FAISS + BM25 + Chroma index.

This enables immediate local development, test running, and benchmark validation.
"""

from __future__ import annotations

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

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.tokenizer import tokenize
from indexing.chunking.strategies import ChunkRecord, ParentRecord, all_chunks_for_parent

SAMPLE_PASSAGES = [
    {
        "query_id": 1,
        "query_type": "description",
        "language_source": "hi",
        "text": "New Delhi is the capital of India and an administrative district of NCT Delhi. It serves as the seat of all three branches of the Government of India, hosting the Rashtrapati Bhavan, Parliament House, and the Supreme Court of India.",
        "alt_lang_text": "नई दिल्ली भारत की राजधानी है।",
    },
    {
        "query_id": 2,
        "query_type": "description",
        "language_source": "ta",
        "text": "Photosynthesis is the biological process by which green plants, algae, and cyanobacteria convert light energy into chemical energy stored in glucose. Chloroplasts contain green chlorophyll pigments that absorb photons from sunlight to power the conversion of carbon dioxide and water into oxygen and sugars.",
        "alt_lang_text": "ஒளிச்சேர்க்கை என்பது தாவரங்கள் சூரிய ஒளியைப் பயன்படுத்தி உணவு தயாரிக்கும் செயல்முறையாகும்.",
    },
    {
        "query_id": 3,
        "query_type": "description",
        "language_source": "bn",
        "text": "Renewable energy is energy derived from natural resources that replenish themselves in less than a human lifetime without depleting the planet's resources. Examples include solar power, wind energy, hydroelectric power, geothermal energy, and biomass.",
        "alt_lang_text": "নবায়নযোগ্য শক্তি হলো এমন শক্তি যা প্রাকৃতিক উৎস থেকে ক্রমাগত পুনরায় পূরণ হয়।",
    },
    {
        "query_id": 4,
        "query_type": "description",
        "language_source": "hi",
        "text": "Vaccines work by introducing a harmless antigen from a weakened or inactivated pathogen into the body. This stimulates the immune system to produce antibodies and memory B-cells and T-cells, providing long-lasting immunity without causing the disease.",
        "alt_lang_text": "টিকা রোগ প্রতিরোধ ক্ষমতা বৃদ্ধি করে।",
    },
    {
        "query_id": 5,
        "query_type": "description",
        "language_source": "ta",
        "text": "Machine learning is a subset of artificial intelligence focused on building applications that learn from data and improve their accuracy over time without being explicitly programmed. Common techniques include supervised learning, unsupervised clustering, and reinforcement learning.",
        "alt_lang_text": "இயந்திரக் கற்றல் என்பது செயற்கை நுண்ணறிவின் ஒரு பகுதியாகும்.",
    },
    {
        "query_id": 6,
        "query_type": "description",
        "language_source": "bn",
        "text": "DNA or deoxyribonucleic acid is the hereditary material in humans and almost all other organisms. It contains the genetic instructions necessary for the development, functioning, growth, and reproduction of all known organisms and many viruses.",
        "alt_lang_text": "ডিএনএ জীবের বংশগত উপাদান।",
    },
    {
        "query_id": 7,
        "query_type": "description",
        "language_source": "hi",
        "text": "Alexander Graham Bell is widely credited with inventing the first practical telephone in 1876. He was awarded the first US patent for the device and subsequently co-founded the American Telephone and Telegraph Company (AT&T).",
        "alt_lang_text": "अलेक्जेंडर ग्राहम बेल ने टेलीफोन का आविष्कार किया था।",
    },
    {
        "query_id": 8,
        "query_type": "description",
        "language_source": "ta",
        "text": "Earthquakes are caused by sudden releases of energy in the Earth's crust that create seismic waves. This usually occurs along geological fault lines where tectonic plates grind against each other or suddenly slip after building stress.",
        "alt_lang_text": "பூகம்பம் என்பது பூமியின் மேலோட்டில் ஏற்படும் திடீர் அதிர்வு ஆகும்.",
    },
    {
        "query_id": 9,
        "query_type": "description",
        "language_source": "bn",
        "text": "The speed of light in a vacuum is approximately 299,792,458 meters per second, commonly denoted as c. In physics, this is considered the universal upper speed limit at which all conventional matter and information can travel.",
        "alt_lang_text": "আলোর গতি প্রতি সেকেন্ডে প্রায় ৩ লক্ষ কিলোমিটার।",
    },
    {
        "query_id": 10,
        "query_type": "description",
        "language_source": "hi",
        "text": "The human heart is a muscular organ that pumps blood through the blood vessels of the circulatory system. Deoxygenated blood is pumped to the lungs to receive oxygen, while oxygenated blood is distributed throughout the rest of the body.",
        "alt_lang_text": "मानव हृदय शरीर में रक्त संचार करता है।",
    },
]


def bootstrap_sample_index() -> None:
    settings = get_settings()
    print("Bootstrapping sample index for local development & test suite...")

    parents: list[ParentRecord] = []
    for idx, item in enumerate(SAMPLE_PASSAGES):
        pid = f"{item['language_source']}:{item['query_id']}:{idx}"
        parents.append(
            ParentRecord(
                parent_id=pid,
                passage_id=pid,
                query_id=item["query_id"],
                query_type=item["query_type"],
                query_cluster=item["query_id"] % 5,
                language_source=item["language_source"],
                text=item["text"],
                alt_lang_text=item["alt_lang_text"],
            )
        )

    # Save parents
    settings.parents_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.parents_path.open("w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    embedder = EmbeddingService(settings.embedding_model)

    def embed_sentences(texts: list[str]) -> np.ndarray:
        return embedder.embed_passages(texts)

    chunks: list[ChunkRecord] = []
    for p in parents:
        chunks.extend(list(all_chunks_for_parent(p, embed_fn=embed_sentences)))

    print(f"Generated {len(chunks)} chunks across 4 strategies.")

    texts = [c.text for c in chunks]
    vectors = embedder.embed_passages(texts)

    # 1. Chroma persistence
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        client.delete_collection("chunks")
    except Exception:
        pass
    collection = client.create_collection(name="chunks", metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c.text for c in chunks],
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
            for c in chunks
        ],
    )

    # 2. FAISS HNSW serving index
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 32
    index.add(vectors)
    settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.faiss_index_path))

    id_map = [c.chunk_id for c in chunks]
    settings.faiss_id_map_path.parent.mkdir(parents=True, exist_ok=True)
    settings.faiss_id_map_path.write_text(json.dumps(id_map), encoding="utf-8")

    # 3. Chunk Parquet
    settings.chunk_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(c) for c in chunks])
    df.to_parquet(settings.chunk_metadata_path, index=False)

    # 4. BM25 Pickle
    settings.bm25_index_dir.mkdir(parents=True, exist_ok=True)
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with (settings.bm25_index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": id_map}, f)

    print(f"Sample index successfully bootstrapped at {settings.faiss_index_path}")


if __name__ == "__main__":
    bootstrap_sample_index()
