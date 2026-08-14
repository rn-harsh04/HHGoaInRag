# Voice RAG — HH Goa 2026 Shortlisting Task 2

A voice-enabled **Retrieval-Augmented Generation (RAG)** pipeline designed for low latency, high grounding precision, and multi-strategy retrieval over the `ai4bharat/MSMARCO-XI` dataset.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://reactjs.org/)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW-blue.svg)](https://github.com/facebookresearch/faiss)
[![FastEmbed](https://img.shields.io/badge/FastEmbed-ONNX-orange.svg)](https://github.com/qdrant/fastembed)
[![Sarvam](https://img.shields.io/badge/STT-Sarvam%20Saaras%20v3-purple.svg)](https://www.sarvam.ai)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%203.1%20Flash%20Lite-4285F4.svg)](https://ai.google.dev)
[![SLA](https://img.shields.io/badge/Retrieval%20SLA-P100%20%E2%89%A4%20200ms-success.svg)](#latency--sla-analytics)

---

## 1. System Architecture

```
User Voice Input (MediaRecorder WebM/WAV, max 25s)
   │
   ▼
POST /v1/query/voice (Multipart FastAPI)
   │
   ├──► ValidateAudioStage (MIME validation, 28s limit, 10MB limit)
   ├──► SarvamSTTStage (Saaras v3 REST API, exponential backoff retries)
   │
   ├──► PreGuardrailStage (Regex filters for unsafe queries, minimum length check)
   │
   ├──► HybridRetrieveStage (STRICT ≤200ms SLA TARGET)
   │      ├── FastEmbed ONNX (BAAI/bge-small-en-v1.5, query embed ~10ms)
   │      ├── Dense Search: In-Memory FAISS HNSWFlat (efSearch=32)
   │      ├── Sparse Search: BM25Okapi (Indic token-aware)
   │      ├── Fusion: Reciprocal Rank Fusion (RRF, k=60)
   │      └── Parent Resolver: Map child hits to full parent passages
   │
   ├──► GeminiGenerateStage (Gemini 3.1 Flash Lite, strict JSON schema output)
   │
   └──► PostGuardrailStage (Token-overlap groundedness test, chunk citation verification)
          │
          ▼
   VoiceQueryResponse JSON ──► React/Vite Glassmorphic Dashboard
```

---

## 2. 4 Chunking Strategies Explained

The dataset is partitioned and indexed using **4 distinct chunking paradigms in a unified vector space**:

| Strategy | Implementation Details | Purpose & Strengths |
| :--- | :--- | :--- |
| **1. Fixed Overlap** | 512 character window, 128 character overlap with sentence boundary snapping. | Baseline windowing with guaranteed boundary continuity. |
| **2. Semantic Splitter** | Sentence tokenization (`.`, `!`, `?`, `।`), cosine similarity check (>0.75) between adjacent embeddings to dynamically group coherent thoughts up to 512 chars. | Prevents splitting semantically unified concepts across arbitrary char boundaries. |
| **3. Metadata-Aware** | Full passage units (or bounded blocks) stamped with structured metadata: `query_id`, `query_type`, `query_cluster`, and `language_source`. | Preserves document-level context and allows query-type and cluster filtering. |
| **4. Parent-Child** | Small 256-char child chunks (64 overlap) indexed for dense vector search, resolving to full parent passage context for LLM generation. | Highly focused vector matching without sacrificing the broader surrounding context for the LLM. |

---

## 3. Strict ≤200ms Latency SLA Architecture

To guarantee the **strict ≤200ms retrieval SLA** in all deployment environments (including Hugging Face Spaces 2 vCPU):

1. **FastEmbed ONNX Singleton**: Embeddings run via quantized ONNX runtime (`bge-small-en-v1.5`), executing in ~8–15ms without PyTorch overhead.
2. **In-Memory FAISS Serving Layer**: Pre-built ChromaDB persistence is loaded into an optimized in-memory `faiss.IndexHNSWFlat` index during startup lifecycle.
3. **Parallel Thread Execution**: Dense FAISS search and Sparse BM25 search execute concurrently in thread pool executors.
4. **Pre-Warmed State**: 5 warm-up queries are executed at startup to eliminate cold-start penalties.

### Benchmark Latency Report (50 Queries × 3 Runs)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) | SLA Status |
| :--- | :---: | :---: | :---: | :---: |
| **Query Embedding (ONNX)** | 11.2 ms | 13.8 ms | 18.5 ms | Passed |
| **Dense FAISS Search** | 14.5 ms | 18.2 ms | 26.0 ms | Passed |
| **BM25 Search** | 12.0 ms | 15.1 ms | 22.4 ms | Passed |
| **RRF Fusion** | 1.1 ms | 1.4 ms | 2.1 ms | Passed |
| **Parent Resolution** | 2.4 ms | 3.1 ms | 4.8 ms | Passed |
| **Total Hybrid Retrieval** | **41.2 ms** | **51.6 ms** | **73.8 ms** | **✓ Passed (&le;200ms)** |

---

## 4. Comprehensive Guardrails & Refusals

The system knows **when NOT to answer**, avoiding hallucinations and inappropriate responses:

* **Pre-Retrieval Safety Filter**: Blocks illegal, hateful, violent, or unsafe prompt injections before any retrieval or LLM computation.
* **Off-Topic Gate**: Evaluates the max cosine similarity from the dense search. If similarity is below `0.42`, the system refuses immediately (`REFUSAL_INSUFFICIENT_INFO`).
* **Post-Generation Groundedness**: Calculates token-overlap between generated answer and retrieved parent contexts (minimum 45% threshold required).
* **Citation Verification**: Ensures all `used_chunk_ids` cited by the LLM strictly belong to the retrieved set.
* **Audit Logging**: All refusal events are appended to `data/logs/refusals.jsonl` with timestamps and stage diagnostics.

---

## 5. Quickstart & Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- Sarvam API Key & Google Gemini API Key

### 1. Clone & Configure Environment
```bash
git clone https://github.com/your-username/HHGOARAG.git
cd HHGOARAG

cp .env.example .env
# Edit .env with your SARVAM_API_KEY and GEMINI_API_KEY
```

### 2. Install Dependencies & Bootstrap Sample Index
```bash
# Backend setup
cd backend
python -m venv .venv
# On Windows: .venv\Scripts\activate | On Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Bootstrap sample index (takes ~10 seconds)
python scripts/bootstrap_sample_data.py

# Frontend setup
cd ../frontend
npm install
```

### 3. Run Development Servers

**On Windows (PowerShell):**
```powershell
# Terminal 1 (Backend):
cd backend
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
# Alternatively without activation:
# .venv\Scripts\python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 (Frontend):
cd frontend
npm run dev
```

**On Linux/macOS:**
```bash
# Terminal 1 (Backend):
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 (Frontend):
cd frontend
npm run dev
```


---

## 6. Running Tests & SLA Benchmark

```bash
cd backend

# Run complete test suite (Chunking, Fusion, Guardrails, Retrieval, Orchestrator)
pytest tests/ -v

# Run the strict latency SLA benchmark gate
python -m benchmarks.run_benchmark --gate 200 --runs 3
```

---

## 7. Hugging Face Spaces Deployment

The repository includes a production-ready **multi-stage Dockerfile** designed specifically for Hugging Face Docker Spaces:

```dockerfile
# Multi-stage builds frontend assets and runs FastAPI on port 7860
docker build -t voice-rag-app .
docker run -p 7860:7860 \
  -e SARVAM_API_KEY="your_key" \
  -e GEMINI_API_KEY="your_key" \
  voice-rag-app
```

Deploying to HF Spaces:
1. Create a new Space on Hugging Face with **Docker** SDK.
2. Push this repository to your Space repo.
3. Set `SARVAM_API_KEY` and `GEMINI_API_KEY` in **Settings > Variables and secrets**.

---

## 8. API Specification

- `GET /health`: Health status, vector count, and SLA target.
- `POST /v1/query/voice`: Multipart voice query with audio file upload (`audio/webm` or `audio/wav`).
- `POST /v1/query/text`: JSON query for testing and benchmarks (`{"query": "...", "language": "en"}`).
- `POST /v1/benchmark/run`: Triggers benchmark harness evaluation.

---

## 9. Submission Checklist (HH Goa 2026)

- [x] **Voice-to-Text**: Sarvam Saaras v3 integrated with exponential backoff.
- [x] **Chunking**: 4 strategies (Fixed, Semantic, Metadata-Aware, Parent-Child).
- [x] **Latency Target**: Strict &le;200ms retrieval SLA verified with benchmark runner.
- [x] **Latency Analytics**: Real-time stage timings & P50/P70/P100 reporting.
- [x] **Harness Orchestration**: Structured pipeline with typed stages and error handling.
- [x] **Guardrails**: Safety filter, off-topic gating, groundedness verification, refusal logging.
- [x] **Single-Container Deployment**: Hugging Face Docker Space compatible.
