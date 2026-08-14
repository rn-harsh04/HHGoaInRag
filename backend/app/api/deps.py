"""FastAPI dependency injection."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.orchestrator.pipeline import VoiceRAGOrchestrator
from app.retrieval.retriever import HybridRetriever
from app.services.embeddings import EmbeddingService
from app.services.gemini_client import GeminiService
from app.services.sarvam_stt import SarvamSTTService


def get_app_state(request: Request):
    return request.app.state


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_orchestrator(request: Request) -> VoiceRAGOrchestrator:
    return request.app.state.orchestrator


SettingsDep = Annotated[Settings, Depends(get_settings)]
RetrieverDep = Annotated[HybridRetriever, Depends(get_retriever)]
OrchestratorDep = Annotated[VoiceRAGOrchestrator, Depends(get_orchestrator)]
