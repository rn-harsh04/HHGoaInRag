"""Health check route."""

from fastapi import APIRouter, Request

from app.schemas.query import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    retriever: object | None = getattr(request.app.state, "retriever", None)
    ready = getattr(request.app.state, "ready", False)
    chunk_count = retriever.chunk_count if retriever else 0
    parent_count = retriever.parent_count if retriever else 0

    return HealthResponse(
        status="ok" if ready else "starting",
        ready=ready,
        index_loaded=retriever is not None,
        chunk_count=chunk_count,
        parent_count=parent_count,
        retrieval_sla_ms=settings.retrieval_budget_ms,
        embedding_model=settings.embedding_model,
    )
