"""Chat and question answering API route for SmartDocs."""

import uuid
from fastapi import APIRouter, HTTPException, status

from backend.rag.rag_pipeline import rag_pipeline
from backend.routes.schemas import ChatRequest, ChatResponse
from backend.utils.exceptions import LLMGenerationError
from backend.utils.logger import get_logger

logger = get_logger("routes_chat")
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question about uploaded documents and receive answer with citations",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Accept user query, retrieve relevant PDF chunks, generate answer with Gemini, and return citations."""
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("Received chat query for session %s: '%s'", session_id, request.query)

    try:
        response = rag_pipeline.generate_answer(
            query=request.query,
            session_id=session_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )

        return ChatResponse(
            answer=response.answer,
            citations=response.citations,
            found_relevant_context=response.found_relevant_context,
            retrieved_chunks_count=response.retrieved_chunks_count,
            session_id=session_id,
        )

    except LLMGenerationError as exc:
        logger.error("LLM Generation error in chat route: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        logger.error("Unhandled error processing chat query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering your question: {exc}",
        ) from exc
