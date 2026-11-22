"""Health check route for backend monitoring."""

from fastapi import APIRouter

from backend.config import settings
from backend.rag.vector_store import vector_store
from backend.routes.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Check backend service health")
async def health_check() -> HealthResponse:
    """Return backend health status, FAISS index metrics, and LLM configuration."""
    stats = vector_store.get_stats()
    gemini_key = settings.GEMINI_API_KEY.strip()
    is_gemini_ready = bool(gemini_key and gemini_key != "your_gemini_api_key_here")

    return HealthResponse(
        status="healthy",
        version=settings.PROJECT_VERSION,
        total_documents=stats["total_documents"],
        total_vectors=stats["total_vectors"],
        gemini_configured=is_gemini_ready,
    )
