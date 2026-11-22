"""SmartDocs - AI-Powered Document Research Assistant Backend Entry Point.

FastAPI application configuring routes, CORS middleware, global exception handlers,
and OpenAPI documentation for the RAG pipeline.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.rag.vector_store import vector_store
from backend.routes import chat, documents, health, upload
from backend.utils.exceptions import SmartDocsException
from backend.utils.logger import get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown hooks."""
    logger.info("Initializing %s v%s...", settings.PROJECT_NAME, settings.PROJECT_VERSION)
    stats = vector_store.get_stats()
    logger.info(
        "Vector store status: %d documents, %d vectors loaded.",
        stats["total_documents"],
        stats["total_vectors"],
    )
    yield
    logger.info("Shutting down %s...", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=(
        "Production-grade Retrieval-Augmented Generation (RAG) backend. "
        "Allows uploading PDF documents, generating Sentence Transformer embeddings, "
        "indexing in FAISS, and retrieving grounded answers with page-level citations using Google Gemini."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SmartDocsException)
async def smartdocs_exception_handler(request: Request, exc: SmartDocsException):
    """Handle custom application exceptions uniformly."""
    logger.warning("Handled domain error: %s (status: %d)", exc.message, exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.__class__.__name__},
    )


# Register Core API Routers
# Direct root routes required by specification:
# POST /upload, POST /chat, GET /documents, DELETE /documents/{id}, DELETE /documents, GET /health
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(chat.router)

# Also expose under /api prefix for standard SaaS frontend API proxying
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(upload.router, prefix=settings.API_PREFIX)
app.include_router(documents.router, prefix=settings.API_PREFIX)
app.include_router(chat.router, prefix=settings.API_PREFIX)


@app.get("/", summary="Root status and API overview")
async def root():
    """Return welcome message, API version, and links to interactive documentation."""
    return {
        "application": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "documents": "/documents",
            "chat": "/chat",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
