"""SmartDocs configuration module.

Loads application settings and hyperparameters from environment variables
with sensible defaults for local and containerized deployments.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Base directory for the backend project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file located at repository root or backend dir
env_path_root = BASE_DIR.parent / ".env"
env_path_backend = BASE_DIR / ".env"

if env_path_root.exists():
    load_dotenv(dotenv_path=env_path_root)
elif env_path_backend.exists():
    load_dotenv(dotenv_path=env_path_backend)
else:
    load_dotenv()


class Settings:
    """Application configuration and hyperparameters."""

    # Project Information
    PROJECT_NAME: str = "SmartDocs - AI Document Research Assistant"
    PROJECT_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Server Configuration
    HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "t")

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Google Gemini LLM Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Sentence Transformers Embedding Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension

    # Document Chunking Hyperparameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "50"))

    # Retrieval Hyperparameters
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))

    # File Storage Paths
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    VECTOR_STORE_DIR: Path = STORAGE_DIR / "vector_store"
    FAISS_INDEX_FILE: Path = VECTOR_STORE_DIR / "smartdocs_faiss.index"
    DOCSTORE_FILE: Path = VECTOR_STORE_DIR / "smartdocs_metadata.json"

    # Supported document formats
    ALLOWED_EXTENSIONS: set = {".pdf"}
    MAX_FILE_SIZE_MB: int = 25

    def __init__(self) -> None:
        """Ensure required storage directories exist on startup."""
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
