"""Sarvam AI Saaras STT client."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.config import Settings
from app.orchestrator.errors import PipelineError, StageName


class SarvamSTTService:
    BASE_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str = "en-IN",
    ) -> tuple[str, float]:
        if not self.settings.sarvam_api_key:
            raise PipelineError(
                StageName.STT,
                "SARVAM_API_KEY not configured",
                http_status=503,
                user_message="Speech service is not configured.",
            )

        headers = {"api-subscription-key": self.settings.sarvam_api_key}
        # Use translate mode for Indic languages so English retrieval index is matched with 0 extra latency
        mode = "transcribe" if (language_code and language_code.startswith("en")) else "translate"
        data = {
            "model": "saaras:v3",
            "mode": mode,
            "language_code": language_code,
        }
        mime = "audio/webm" if (filename and filename.endswith(".webm")) else "audio/wav"
        files = {"file": (filename or "recording.wav", audio_bytes, mime)}

        last_error: Exception | None = None
        for attempt in range(self.settings.stt_max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.settings.stt_timeout_sec) as client:
                    import time

                    start = time.perf_counter()
                    resp = await client.post(self.BASE_URL, headers=headers, data=data, files=files)
                    latency_ms = (time.perf_counter() - start) * 1000

                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = RuntimeError(f"Sarvam HTTP {resp.status_code}: {resp.text[:200]}")
                    await asyncio.sleep(2**attempt)
                    continue

                if resp.status_code >= 400:
                    raise PipelineError(
                        StageName.STT,
                        resp.text,
                        http_status=502,
                        user_message="Couldn't transcribe audio. Try again.",
                    )

                payload = resp.json()
                text = (
                    payload.get("transcript")
                    or payload.get("text")
                    or payload.get("output")
                    or ""
                ).strip()
                if not text:
                    raise PipelineError(
                        StageName.STT,
                        f"Empty transcript: {payload}",
                        http_status=502,
                        user_message="Couldn't transcribe audio. Try again.",
                    )
                return text, latency_ms

            except PipelineError:
                raise
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(2**attempt)

        raise PipelineError(
            StageName.STT,
            str(last_error),
            http_status=502,
            user_message="Couldn't transcribe audio. Try again.",
        )
