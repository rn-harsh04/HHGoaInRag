"""Audio upload validation."""

from __future__ import annotations

import io
import struct
import wave

from app.config import Settings
from app.orchestrator.errors import PipelineError, StageName


ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/mpeg",
    "audio/ogg",
    "application/octet-stream",
}


def _wav_duration_sec(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


def validate_audio(
    data: bytes,
    filename: str,
    content_type: str | None,
    settings: Settings,
) -> tuple[bytes, float | None]:
    if not data:
        raise PipelineError(
            StageName.VALIDATE_AUDIO,
            "Empty audio",
            http_status=400,
            user_message="No audio received. Please record again.",
        )

    if len(data) > settings.max_audio_bytes:
        raise PipelineError(
            StageName.VALIDATE_AUDIO,
            "Audio too large",
            http_status=400,
            user_message="Audio file too large. Please record a shorter clip.",
        )

    # Extract base MIME type (e.g. "audio/webm;codecs=opus" -> "audio/webm")
    base_content_type = content_type.split(";")[0].strip().lower() if content_type else ""

    if base_content_type and not (
        base_content_type.startswith("audio/")
        or base_content_type in ("application/octet-stream", "video/webm", "video/ogg")
    ):
        raise PipelineError(
            StageName.VALIDATE_AUDIO,
            f"Unsupported content type: {content_type}",
            http_status=400,
            user_message="Unsupported audio format. Use WAV or WebM.",
        )

    duration = _wav_duration_sec(data)
    if duration is not None and duration > settings.max_audio_duration_sec:
        raise PipelineError(
            StageName.VALIDATE_AUDIO,
            f"Audio too long: {duration}s",
            http_status=400,
            user_message="Please record a shorter clip (under 30 seconds).",
        )

    return data, duration
