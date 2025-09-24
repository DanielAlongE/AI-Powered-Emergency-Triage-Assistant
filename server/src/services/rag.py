from __future__ import annotations
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from config import get_settings
from logger import get_logger

logger = get_logger(__name__)


class ProtocolRAG:
    def __init__(self) -> None:
        settings = get_settings()
        self._persist_dir = str(settings.chroma_db_path)
        self._embedding_model = settings.embedding_model
        self._client = self._create_vector_store()
        self._retriever = self._client.as_retriever(search_kwargs={"k": 4})
        logger.info(
            "rag_initialized", directory=self._persist_dir, embedding_model=self._embedding_model
        )

    def _create_vector_store(self) -> Chroma:
        embeddings = OpenAIEmbeddings(
            model=self._embedding_model,
            openai_api_key=get_settings().openai_api_key,
        )
        return Chroma(persist_directory=self._persist_dir, embedding_function=embeddings)

    @property
    def retriever(self) -> VectorStoreRetriever:
        return self._retriever

    def query(self, query: str, *, k: int = 4) -> List[Document]:
        docs = self._retriever.get_relevant_documents(query)
        if docs:
            logger.info("rag_hits", query=query, results=len(docs))
        else:
            logger.warning("rag_no_results", query=query)
        return docs[:k]


_protocol_rag: ProtocolRAG | None = None


def get_protocol_rag() -> ProtocolRAG:
    global _protocol_rag
    if _protocol_rag is None:
        _protocol_rag = ProtocolRAG()
    return _protocol_rag