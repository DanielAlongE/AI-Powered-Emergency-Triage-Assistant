"""
Ollama embeddings service wrapper for local embedding generation.
Uses langchain-ollama for integration with ChromaDB and RAG.
"""

from __future__ import annotations
from typing import List
from langchain_ollama import OllamaEmbeddings
from config import get_settings
from logger import get_logger

logger = get_logger(__name__)


class OllamaEmbeddingService:
    """
    Wrapper for Ollama embeddings using langchain-ollama.
    Provides a consistent interface for generating embeddings with local Ollama models.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        """
        Initialize Ollama embeddings service.

        Args:
            model: Embedding model name (defaults to config setting)
            base_url: Ollama server URL (defaults to config setting)
        """
        settings = get_settings()

        self._model = model or settings.ollama_embedding_model
        self._base_url = base_url or settings.ollama_host

        # Initialize the LangChain Ollama embeddings
        self._embeddings = OllamaEmbeddings(
            model=self._model,
            base_url=self._base_url
        )

        logger.info("ollama_embeddings_initialized",
                   model=self._model,
                   base_url=self._base_url)

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Get the underlying LangChain OllamaEmbeddings instance."""
        return self._embeddings

    @property
    def model(self) -> str:
        """Get the embedding model name."""
        return self._model

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding vector
        """
        try:
            embedding = self._embeddings.embed_query(text)
            logger.debug("ollama_query_embedded",
                        text_length=len(text),
                        embedding_dim=len(embedding))
            return embedding
        except Exception as e:
            logger.error("ollama_embed_query_failed",
                        error=str(e),
                        model=self._model)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        try:
            embeddings = self._embeddings.embed_documents(texts)
            logger.info("ollama_documents_embedded",
                       document_count=len(texts),
                       embedding_dim=len(embeddings[0]) if embeddings else 0,
                       model=self._model)
            return embeddings
        except Exception as e:
            logger.error("ollama_embed_documents_failed",
                        error=str(e),
                        document_count=len(texts),
                        model=self._model)
            raise

    def check_availability(self) -> bool:
        """
        Check if Ollama server is available and the embedding model is accessible.

        Returns:
            True if available, False otherwise
        """
        try:
            # Try a simple embedding to test availability
            test_embedding = self.embed_query("test")
            logger.info("ollama_embeddings_available", model=self._model)
            return len(test_embedding) > 0
        except Exception as e:
            logger.warning("ollama_embeddings_unavailable",
                          error=str(e),
                          model=self._model)
            return False


# Singleton pattern for easy access
_ollama_embedding_service: OllamaEmbeddingService | None = None


def get_ollama_embedding_service(model: str | None = None, base_url: str | None = None) -> OllamaEmbeddingService:
    """
    Get singleton Ollama embedding service instance.

    Args:
        model: Embedding model name (defaults to config)
        base_url: Ollama server URL (defaults to config)

    Returns:
        OllamaEmbeddingService instance
    """
    global _ollama_embedding_service
    if _ollama_embedding_service is None:
        _ollama_embedding_service = OllamaEmbeddingService(model=model, base_url=base_url)
    return _ollama_embedding_service


def is_ollama_embeddings_available() -> bool:
    """
    Quick check if Ollama embeddings are available.

    Returns:
        True if Ollama embeddings can be used, False otherwise
    """
    try:
        service = get_ollama_embedding_service()
        return service.check_availability()
    except Exception:
        return False