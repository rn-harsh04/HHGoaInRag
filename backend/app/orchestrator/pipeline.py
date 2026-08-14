"""Voice RAG pipeline orchestrator."""

from __future__ import annotations

import uuid

from app.config import Settings
from app.guardrails.groundedness import token_overlap_score, validate_chunk_ids
from app.guardrails.refusal_logger import log_refusal
from app.guardrails.safety import is_off_topic, is_unsafe
from app.logging.timing import PipelineTimings, timed_stage
from app.orchestrator.errors import PipelineError, StageName
from app.orchestrator.stages.validate_audio import validate_audio
from app.retrieval.retriever import HybridRetriever
from app.schemas.query import (
    AnswerPayload,
    GroundednessPayload,
    ResponseStatus,
    RetrievalPayload,
    TextQueryRequest,
    TimingsPayload,
    TranscriptResult,
    VoiceQueryResponse,
)
from app.services.gemini_client import GeminiService
from app.services.sarvam_stt import SarvamSTTService

INSUFFICIENT_MSG = (
    "I don't have enough information in the indexed passages to answer that question."
)


class VoiceRAGOrchestrator:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        stt: SarvamSTTService,
        gemini: GeminiService,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.stt = stt
        self.gemini = gemini

    def _refusal_response(
        self,
        request_id: str,
        status: ResponseStatus,
        message: str,
        *,
        transcript: TranscriptResult | None = None,
        retrieval: RetrievalPayload | None = None,
        timings: PipelineTimings | None = None,
    ) -> VoiceQueryResponse:
        return VoiceQueryResponse(
            status=status,
            transcript=transcript,
            retrieval=retrieval,
            answer=AnswerPayload(text=message, refused=True, refusal_reason=message),
            groundedness=GroundednessPayload(score=0.0, passed=False),
            timings_ms=TimingsPayload(
                stt=timings.stt if timings else 0,
                pre_guardrail=timings.pre_guardrail if timings else 0,
                retrieval=timings.retrieval if timings else 0,
                llm=timings.llm if timings else 0,
                post_guardrail=timings.post_guardrail if timings else 0,
                total=timings.total if timings else 0,
            ),
            request_id=request_id,
            message=message,
        )

    async def run_text_query(self, body: TextQueryRequest) -> VoiceQueryResponse:
        return await self._run_pipeline(
            query=body.query.strip(),
            request_id=str(uuid.uuid4()),
            transcript=None,
        )

    async def run_voice_query(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str | None,
        language_code: str = "en-IN",
    ) -> VoiceQueryResponse:
        request_id = str(uuid.uuid4())
        timings = PipelineTimings()

        validate_audio(audio_bytes, filename, content_type, self.settings)

        with timed_stage(timings, "stt"):
            text, stt_ms = await self.stt.transcribe(audio_bytes, filename, language_code)
        timings.stt = stt_ms

        transcript = TranscriptResult(text=text, language=language_code, stt_latency_ms=stt_ms)
        return await self._run_pipeline(query=text, request_id=request_id, transcript=transcript, timings=timings)

    async def _run_pipeline(
        self,
        query: str,
        request_id: str,
        transcript: TranscriptResult | None,
        timings: PipelineTimings | None = None,
    ) -> VoiceQueryResponse:
        timings = timings or PipelineTimings()

        with timed_stage(timings, "pre_guardrail"):
            unsafe, reason = is_unsafe(query)
        if unsafe:
            if self.settings.log_refusals:
                log_refusal(
                    self.settings.refusal_log_path,
                    request_id=request_id,
                    query=query,
                    reason=reason or "unsafe",
                    stage="pre_guardrail",
                )
            return self._refusal_response(
                request_id,
                ResponseStatus.REFUSAL_UNSAFE,
                reason or "Unsafe input",
                transcript=transcript,
                timings=timings,
            )

        with timed_stage(timings, "retrieval"):
            retrieval, _stats = await self.retriever.retrieve(query)
        timings.retrieval = retrieval.latency_ms

        if not retrieval.chunks or is_off_topic(retrieval.max_score, self.settings.retrieval_min_score):
            if self.settings.log_refusals:
                log_refusal(
                    self.settings.refusal_log_path,
                    request_id=request_id,
                    query=query,
                    reason="off_topic_or_empty",
                    stage="retrieve",
                    max_score=retrieval.max_score,
                )
            return self._refusal_response(
                request_id,
                ResponseStatus.REFUSAL_INSUFFICIENT_INFO,
                INSUFFICIENT_MSG,
                transcript=transcript,
                retrieval=retrieval,
                timings=timings,
            )

        chunk_dicts = [
            {"chunk_id": c.chunk_id, "strategy": c.strategy, "text": c.text}
            for c in retrieval.chunks
        ]

        try:
            with timed_stage(timings, "llm"):
                llm_result, llm_ms = await self.gemini.generate(
                    query, retrieval.parents_used, chunk_dicts
                )
            timings.llm = llm_ms
        except PipelineError as exc:
            return VoiceQueryResponse(
                status=ResponseStatus.ERROR,
                transcript=transcript,
                retrieval=retrieval,
                timings_ms=TimingsPayload(
                    stt=timings.stt,
                    pre_guardrail=timings.pre_guardrail,
                    retrieval=timings.retrieval,
                    total=timings.total,
                ),
                request_id=request_id,
                message=exc.user_message,
            )

        retrieved_ids = {c.chunk_id for c in retrieval.chunks}
        context_texts = [p.text for p in retrieval.parents_used] + [c.text for c in retrieval.chunks]

        with timed_stage(timings, "post_guardrail"):
            overlap = token_overlap_score(llm_result.answer, context_texts)
            ids_valid = validate_chunk_ids(llm_result.used_chunk_ids, retrieved_ids)
            grounded_pass = (
                not llm_result.refused
                and overlap >= self.settings.groundedness_min_score
                and ids_valid
            )

        groundedness = GroundednessPayload(score=overlap, passed=grounded_pass, method="token_overlap")

        if llm_result.refused:
            if self.settings.log_refusals:
                log_refusal(
                    self.settings.refusal_log_path,
                    request_id=request_id,
                    query=query,
                    reason=llm_result.refusal_reason or "model_refused",
                    stage="generate",
                    max_score=retrieval.max_score,
                )
            return self._refusal_response(
                request_id,
                ResponseStatus.REFUSAL_INSUFFICIENT_INFO,
                llm_result.refusal_reason or INSUFFICIENT_MSG,
                transcript=transcript,
                retrieval=retrieval,
                timings=timings,
            )

        if not grounded_pass:
            if self.settings.log_refusals:
                log_refusal(
                    self.settings.refusal_log_path,
                    request_id=request_id,
                    query=query,
                    reason="groundedness_failed",
                    stage="post_guardrail",
                    max_score=retrieval.max_score,
                )
            return self._refusal_response(
                request_id,
                ResponseStatus.REFUSAL_INSUFFICIENT_INFO,
                INSUFFICIENT_MSG,
                transcript=transcript,
                retrieval=retrieval,
                timings=timings,
            )

        return VoiceQueryResponse(
            status=ResponseStatus.SUCCESS,
            transcript=transcript,
            retrieval=retrieval,
            answer=AnswerPayload(
                text=llm_result.answer,
                confidence=llm_result.confidence,
                used_chunk_ids=llm_result.used_chunk_ids,
                refused=False,
            ),
            groundedness=groundedness,
            timings_ms=TimingsPayload(
                stt=timings.stt,
                pre_guardrail=timings.pre_guardrail,
                retrieval=timings.retrieval,
                llm=timings.llm,
                post_guardrail=timings.post_guardrail,
                total=timings.total,
            ),
            request_id=request_id,
        )
