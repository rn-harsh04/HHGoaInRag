"""Pydantic schemas for API requests and responses."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    REFUSAL_INSUFFICIENT_INFO = "refusal_insufficient_info"
    REFUSAL_UNSAFE = "refusal_unsafe"
    ERROR = "error"


class TranscriptResult(BaseModel):
    text: str
    language: str = "en-IN"
    stt_latency_ms: float = 0.0


class ChunkHit(BaseModel):
    chunk_id: str
    strategy: str
    text: str
    score: float
    parent_id: str
    passage_id: str
    rank: int | None = None


class ParentContext(BaseModel):
    parent_id: str
    text: str
    passage_id: str
    language_source: str = "en"


class RetrievalPayload(BaseModel):
    chunks: list[ChunkHit] = Field(default_factory=list)
    parents_used: list[ParentContext] = Field(default_factory=list)
    max_score: float = 0.0
    latency_ms: float = 0.0
    timed_out: bool = False


class AnswerPayload(BaseModel):
    text: str = ""
    confidence: float = 0.0
    used_chunk_ids: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


class GroundednessPayload(BaseModel):
    score: float = 0.0
    passed: bool = False
    method: str = "token_overlap"


class TimingsPayload(BaseModel):
    stt: float = 0.0
    pre_guardrail: float = 0.0
    retrieval: float = 0.0
    llm: float = 0.0
    post_guardrail: float = 0.0
    total: float = 0.0


class VoiceQueryResponse(BaseModel):
    status: ResponseStatus
    transcript: TranscriptResult | None = None
    retrieval: RetrievalPayload | None = None
    answer: AnswerPayload | None = None
    groundedness: GroundednessPayload | None = None
    timings_ms: TimingsPayload = Field(default_factory=TimingsPayload)
    request_id: str
    message: str | None = None


class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    language: str = "en"


class HealthResponse(BaseModel):
    status: str
    ready: bool
    index_loaded: bool
    chunk_count: int = 0
    parent_count: int = 0
    retrieval_sla_ms: int = 200
    embedding_model: str = ""


class BenchmarkStageStats(BaseModel):
    p50: float
    p70: float
    p100: float


class BenchmarkReport(BaseModel):
    retrieval_total_ms: BenchmarkStageStats
    embed_query_ms: BenchmarkStageStats
    dense_search_ms: BenchmarkStageStats
    bm25_search_ms: BenchmarkStageStats
    fusion_ms: BenchmarkStageStats
    parent_resolve_ms: BenchmarkStageStats
    sla_target_ms: int = 200
    sla_pass: bool
    query_count: int
    runs: int
    stages: dict[str, Any] = Field(default_factory=dict)
