"""RAG Pipeline Orchestrator for SmartDocs.

Combines semantic retrieval via FAISS with Google Gemini API to produce grounded,
strictly contextual answers accompanied by verified source citations.
"""

from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.rag.embeddings import EmbeddingManager, embedding_manager
from backend.rag.models import Citation, RAGResponse, SearchResult
from backend.rag.vector_store import FAISSVectorStore, vector_store
from backend.utils.exceptions import LLMGenerationError
from backend.utils.logger import get_logger

logger = get_logger("rag_pipeline")

SYSTEM_PROMPT = """You are SmartDocs, an expert AI Document Research Assistant.
Your task is to answer the user's question STRICTLY and ACCURATELY using ONLY the provided document excerpts below.

CRITICAL INSTRUCTIONS:
1. Ground your answer completely in the provided context. Do NOT use outside knowledge or make assumptions.
2. If the provided excerpts do not contain sufficient information to answer the question, state clearly:
   "I could not find information regarding this question in the uploaded documents."
3. When referencing information, cite the relevant sources in brackets like [Source 1], [Source 2], etc.
4. Maintain a professional, objective, and well-structured tone with markdown formatting (bullet points, bold text).
"""


class SessionHistoryManager:
    """Manages in-memory conversational history per session."""

    def __init__(self, max_history_turns: int = 10) -> None:
        self.max_turns = max_history_turns
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation turns for a session."""
        return self._sessions.get(session_id, [])

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Citation]] = None,
    ) -> None:
        """Add a conversation turn to the session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        turn: Dict[str, Any] = {"role": role, "content": content}
        if citations:
            turn["citations"] = [c.model_dump() for c in citations]

        self._sessions[session_id].append(turn)

        # Trim old history if exceeding maximum turns
        if len(self._sessions[session_id]) > self.max_turns * 2:
            self._sessions[session_id] = self._sessions[session_id][-self.max_turns * 2 :]

    def clear_session(self, session_id: str) -> None:
        """Clear history for a specific session."""
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear all active sessions."""
        self._sessions.clear()


class RAGPipeline:
    """Orchestrates retrieval, context construction, and LLM generation."""

    def __init__(
        self,
        v_store: Optional[FAISSVectorStore] = None,
        embed_mgr: Optional[EmbeddingManager] = None,
    ) -> None:
        self.vector_store = v_store or vector_store
        self.embedding_manager = embed_mgr or embedding_manager
        self.history_manager = SessionHistoryManager()
        self._gemini_client = None

    def _get_gemini_client(self):
        """Lazy-initialize Google Gemini client."""
        if self._gemini_client is None:
            api_key = settings.GEMINI_API_KEY.strip()
            if not api_key or api_key == "your_gemini_api_key_here":
                logger.warning("Gemini API key is not set or using placeholder value.")
                return None

            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=api_key)
                logger.info("Initialized Google Gemini client with model: %s", settings.GEMINI_MODEL)
            except Exception as exc:
                logger.error("Failed to initialize Google Gemini client: %s", exc)
                raise LLMGenerationError(f"Gemini client initialization failed: {exc}") from exc

        return self._gemini_client

    def build_context(self, search_results: List[SearchResult]) -> Tuple[str, List[Citation]]:
        """Construct prompt context string and structured citation models from search results.

        Args:
            search_results: List of SearchResult models from vector store.

        Returns:
            Tuple of (formatted_context_string, list_of_citations).
        """
        context_parts: List[str] = []
        citations: List[Citation] = []

        for i, result in enumerate(search_results, start=1):
            chunk = result.chunk
            context_header = f"[Source {i}: {chunk.filename} | Page {chunk.page_number}]"
            context_parts.append(f"{context_header}\n{chunk.text}\n")

            citations.append(
                Citation(
                    citation_index=i,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    snippet=chunk.text,
                    relevance_score=round(result.score, 4),
                )
            )

        formatted_context = "\n---\n".join(context_parts)
        return formatted_context, citations

    def generate_answer(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """Process user query through semantic retrieval and LLM generation.

        Args:
            query: The user's input question.
            session_id: Optional session ID for conversation history.
            top_k: Number of chunks to retrieve.
            similarity_threshold: Minimum similarity score.

        Returns:
            RAGResponse object with answer and verified citations.
        """
        clean_query = query.strip()
        if not clean_query:
            return RAGResponse(
                answer="Please enter a question to research your documents.",
                citations=[],
                found_relevant_context=False,
                retrieved_chunks_count=0,
            )

        # Check if vector store has any indexed documents
        stats = self.vector_store.get_stats()
        if stats["total_vectors"] == 0:
            return RAGResponse(
                answer=(
                    "No documents have been uploaded yet. "
                    "Please upload one or more PDF files using the sidebar to begin asking questions."
                ),
                citations=[],
                found_relevant_context=False,
                retrieved_chunks_count=0,
            )

        # 1. Generate query embedding
        logger.info("Generating query embedding for: '%s'", clean_query)
        query_embedding = self.embedding_manager.embed_query(clean_query)

        # 2. Retrieve top-k semantic matches
        search_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k or settings.TOP_K_RETRIEVAL,
            similarity_threshold=similarity_threshold,
        )

        if not search_results:
            logger.info("No matching chunks met the similarity threshold for query: '%s'", clean_query)
            return RAGResponse(
                answer="I could not find information regarding this question in the uploaded documents.",
                citations=[],
                found_relevant_context=False,
                retrieved_chunks_count=0,
            )

        # 3. Format context and citations
        context_text, citations = self.build_context(search_results)

        # 4. Generate LLM response
        client = self._get_gemini_client()
        if client is None:
            # Fallback if Gemini API key is not configured
            answer_text = (
                "**Retrieved Context Excerpt:**\n\n"
                f"{search_results[0].chunk.text}\n\n"
                "*(Note: Gemini API key is not configured in `.env`. "
                "Retrieved top matching excerpts directly from FAISS index. "
                "Configure `GEMINI_API_KEY` for synthesized answers.)*"
            )
        else:
            try:
                # Include previous conversation history if available
                history_prompt = ""
                if session_id:
                    history = self.history_manager.get_history(session_id)
                    if history:
                        history_lines = [
                            f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
                            for h in history[-4:]
                        ]
                        history_prompt = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

                prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"{history_prompt}"
                    f"Document Excerpts:\n{context_text}\n\n"
                    f"User Question: {clean_query}\n\n"
                    f"Synthesized Answer:"
                )

                logger.info("Calling Gemini API (%s)...", settings.GEMINI_MODEL)
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                )
                answer_text = response.text or "No answer could be generated."

            except Exception as exc:
                logger.error("Gemini API call failed: %s", exc)
                raise LLMGenerationError(f"Failed to generate answer from Gemini: {exc}") from exc

        # Update session conversation history
        if session_id:
            self.history_manager.add_turn(session_id, "user", clean_query)
            self.history_manager.add_turn(session_id, "assistant", answer_text, citations)

        return RAGResponse(
            answer=answer_text,
            citations=citations,
            found_relevant_context=True,
            retrieved_chunks_count=len(search_results),
        )


# Global singleton pipeline instance
rag_pipeline = RAGPipeline()
