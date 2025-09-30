"""
Ollama-based RAG service for emergency protocol retrieval.
Uses Ollama embeddings (nomic-embed-text) with ChromaDB.
"""

from __future__ import annotations
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from services.ollama_embeddings import get_ollama_embedding_service
from config import get_settings
from logger import get_logger

logger = get_logger(__name__)


class ProtocolRAGOllama:
    """
    Protocol RAG service using Ollama embeddings.
    Provides retrieval-augmented generation for emergency protocol queries.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._persist_dir = str(settings.chroma_ollama_path)
        self._embedding_model = settings.ollama_embedding_model

        # Get Ollama embedding service
        self._embedding_service = get_ollama_embedding_service()

        # Create vector store with Ollama embeddings
        self._client = self._create_vector_store()
        self._retriever = self._client.as_retriever(search_kwargs={"k": 8})

        logger.info(
            "ollama_rag_initialized",
            directory=self._persist_dir,
            embedding_model=self._embedding_model
        )

    def _create_vector_store(self) -> Chroma:
        """Create ChromaDB vector store with Ollama embeddings."""
        try:
            return Chroma(
                persist_directory=self._persist_dir,
                embedding_function=self._embedding_service.embeddings
            )
        except Exception as e:
            logger.error("ollama_vector_store_creation_failed",
                        error=str(e),
                        persist_dir=self._persist_dir)
            raise

    @property
    def retriever(self) -> VectorStoreRetriever:
        """Get the vector store retriever."""
        return self._retriever

    @property
    def embedding_model(self) -> str:
        """Get the embedding model name."""
        return self._embedding_model

    @property
    def persist_directory(self) -> str:
        """Get the ChromaDB persistence directory."""
        return self._persist_dir

    def query(self, query: str, *, k: int = 8) -> List[Document]:
        """
        Query the vector database for relevant documents.

        Args:
            query: Search query text
            k: Number of documents to retrieve

        Returns:
            List of relevant documents from the protocol database
        """
        try:
            docs = self._retriever.get_relevant_documents(query)
            if docs:
                logger.info("ollama_rag_hits",
                           query=query,
                           results=len(docs),
                           embedding_model=self._embedding_model)
            else:
                logger.warning("ollama_rag_no_results",
                              query=query,
                              embedding_model=self._embedding_model)
            return docs[:k]
        except Exception as e:
            logger.error("ollama_rag_query_failed",
                        error=str(e),
                        query=query)
            return []

    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """
        Perform similarity search on the vector database.

        Args:
            query: Search query text
            k: Number of similar documents to return

        Returns:
            List of similar documents
        """
        try:
            docs = self._client.similarity_search(query, k=k)
            logger.info("ollama_similarity_search",
                       query=query,
                       results=len(docs),
                       embedding_model=self._embedding_model)
            return docs
        except Exception as e:
            logger.error("ollama_similarity_search_failed",
                        error=str(e),
                        query=query)
            return []

    def check_availability(self) -> bool:
        """
        Check if the Ollama RAG service is available.

        Returns:
            True if the service is working, False otherwise
        """
        try:
            # Test with a simple query
            test_docs = self.similarity_search("test query", k=1)
            logger.info("ollama_rag_availability_check",
                       available=True,
                       test_results=len(test_docs))
            return True
        except Exception as e:
            logger.warning("ollama_rag_unavailable",
                          error=str(e),
                          persist_dir=self._persist_dir)
            return False

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector database.

        Returns:
            Number of documents, or -1 if unable to determine
        """
        try:
            # Get collection info
            collection = self._client._collection
            count = collection.count()
            logger.info("ollama_rag_document_count", count=count)
            return count
        except Exception as e:
            logger.warning("ollama_rag_count_failed", error=str(e))
            return -1


# Singleton pattern for easy access
_protocol_rag_ollama: ProtocolRAGOllama | None = None


def get_protocol_rag_ollama() -> ProtocolRAGOllama:
    """
    Get singleton Ollama protocol RAG instance.

    Returns:
        ProtocolRAGOllama instance
    """
    global _protocol_rag_ollama
    if _protocol_rag_ollama is None:
        _protocol_rag_ollama = ProtocolRAGOllama()
    return _protocol_rag_ollama


def is_ollama_rag_available() -> bool:
    """
    Quick check if Ollama RAG is available and has documents.

    Returns:
        True if Ollama RAG is ready to use, False otherwise
    """
    try:
        rag = get_protocol_rag_ollama()
        return rag.check_availability() and rag.get_document_count() > 0
    except Exception:
        return False