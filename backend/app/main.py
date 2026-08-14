"""FastAPI application entrypoint."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.routes import benchmark, health, query
from app.config import get_settings
from app.orchestrator.pipeline import VoiceRAGOrchestrator
from app.retrieval.retriever import HybridRetriever
from app.services.embeddings import EmbeddingService
from app.services.gemini_client import GeminiService
from app.services.sarvam_stt import SarvamSTTService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.ready = False

    embedder = EmbeddingService.get_instance(settings)
    retriever = HybridRetriever.load(settings, embedder)
    embedder.warmup(settings.warmup_queries)

    stt = SarvamSTTService(settings)
    gemini = GeminiService(settings)
    orchestrator = VoiceRAGOrchestrator(settings, retriever, stt, gemini)

    app.state.retriever = retriever
    app.state.orchestrator = orchestrator
    app.state.ready = True

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="HHGOARAG Voice RAG", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(benchmark.router)

    static_dir = BACKEND_ROOT.parent / "frontend" / "dist"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
