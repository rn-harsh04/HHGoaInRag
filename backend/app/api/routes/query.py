"""Query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from app.api.deps import OrchestratorDep
from app.orchestrator.errors import PipelineError
from app.schemas.query import ResponseStatus, TextQueryRequest, VoiceQueryResponse

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query/voice", response_model=VoiceQueryResponse)
async def query_voice(
    orchestrator: OrchestratorDep,
    audio: UploadFile = File(...),
    language_hint: str = Form(default="en-IN"),
) -> VoiceQueryResponse:
    data = await audio.read()
    try:
        return await orchestrator.run_voice_query(
            audio_bytes=data,
            filename=audio.filename or "audio.wav",
            content_type=audio.content_type,
            language_code=language_hint or "en-IN",
        )
    except PipelineError as exc:
        return VoiceQueryResponse(
            status=ResponseStatus.ERROR,
            request_id="error",
            message=exc.user_message,
        )


@router.post("/query/text", response_model=VoiceQueryResponse)
async def query_text(
    orchestrator: OrchestratorDep,
    body: TextQueryRequest,
) -> VoiceQueryResponse:
    return await orchestrator.run_text_query(body)
