"""Gemini structured answer generation."""

from __future__ import annotations

import json
import time
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.config import Settings
from app.orchestrator.errors import PipelineError, StageName
from app.schemas.query import ParentContext


class GeminiAnswerSchema(BaseModel):
    answer: str = ""
    used_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    refusal_reason: str | None = None


SYSTEM_PROMPT = """You are a grounded QA assistant. Answer ONLY using the provided passages.
If passages do not contain enough information, set refused=true and explain in refusal_reason.
Respond in English. Return valid JSON matching the schema.
Cite chunk IDs you used in used_chunk_ids. Do not hallucinate facts not in passages."""


class GeminiService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.settings.gemini_api_key:
                raise PipelineError(
                    StageName.GENERATE,
                    "GEMINI_API_KEY not configured",
                    http_status=503,
                    user_message="Answer generation is not configured.",
                )
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def _build_prompt(
        self,
        query: str,
        parents: list[ParentContext],
        chunks: list[dict[str, Any]],
    ) -> str:
        context_parts = []
        for p in parents:
            context_parts.append(f"[PARENT {p.parent_id}]\n{p.text}")
        for c in chunks:
            context_parts.append(f"[CHUNK {c['chunk_id']} | {c['strategy']}]\n{c['text']}")
        context = "\n\n".join(context_parts)
        return f"Question: {query}\n\nContext:\n{context}"

    async def generate(
        self,
        query: str,
        parents: list[ParentContext],
        chunks: list[dict[str, Any]],
    ) -> tuple[GeminiAnswerSchema, float]:
        prompt = self._build_prompt(query, parents, chunks)

        start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=GeminiAnswerSchema.model_json_schema(),
                    temperature=0.2,
                ),
            )
        except Exception as exc:
            raise PipelineError(
                StageName.GENERATE,
                str(exc),
                http_status=503,
                user_message="Answer generation failed. Try again.",
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.text or "{}"

        try:
            parsed = GeminiAnswerSchema.model_validate(json.loads(raw))
        except Exception:
            # One retry with stricter instruction handled by caller if needed
            raise PipelineError(
                StageName.GENERATE,
                f"Invalid JSON from Gemini: {raw[:300]}",
                http_status=502,
                user_message="Answer generation failed. Try again.",
            )

        return parsed, latency_ms
