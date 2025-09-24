#!/usr/bin/env python3
"""
Script to ingest ESI protocol content into ChromaDB vector database.
Based on the original implementation from project/scripts/ingest_protocol.py
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List

# Ensure we can import from our source modules
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import get_settings
from logger import configure_logging, get_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

logger = get_logger(__name__)


def load_protocol_text(path: Path) -> str:
    """
    Load protocol text from file, supporting both PDF and text formats.
    """
    if not path.exists():
        raise FileNotFoundError(f"ESI source not found at {path}")

    if path.suffix.lower() == '.pdf':
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with: poetry add pypdf")

        reader = pypdf.PdfReader(str(path))
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text + "\n"

        if not text.strip():
            raise ValueError(f"No text could be extracted from PDF: {path}")

        logger.info("pdf_loaded", pages=len(reader.pages), text_length=len(text))
        return text
    else:
        # Handle markdown, txt, and other text files
        content = path.read_text(encoding="utf-8")
        logger.info("text_file_loaded",
                   file_type=path.suffix,
                   text_length=len(content))
        return content


def create_document_chunks(text: str, source_path: Path) -> List[Document]:
    """
    Split text into chunks and create Document objects.
    """
    # Use settings similar to the original implementation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )

    documents = splitter.create_documents(
        [text],
        metadatas=[{"source": str(source_path)}]
    )

    logger.info("text_split",
                total_chunks=len(documents),
                chunk_size=600,
                chunk_overlap=100)

    # Log first few chunks for debugging
    for i, doc in enumerate(documents[:3]):
        logger.info(f"chunk_{i}",
                   length=len(doc.page_content),
                   preview=doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content)

    return documents


def main():
    """
    Main ingestion function - loads protocol content into ChromaDB.
    """
    settings = get_settings()

    # Configure logging
    configure_logging(settings.log_level)

    logger.info("ingest_start",
                source_path=str(settings.esi_source_path),
                chroma_path=str(settings.chroma_db_path))

    # Validate OpenAI API key
    if not settings.openai_api_key:
        logger.error("openai_api_key_missing",
                    message="OpenAI API key is required for embeddings")
        return 1

    try:
        # Load and log the source text
        text = load_protocol_text(settings.esi_source_path)
        logger.info("text_loaded",
                    source=str(settings.esi_source_path),
                    text_length=len(text),
                    text_preview=text[:200] + "..." if len(text) > 200 else text)

        # Create document chunks
        documents = create_document_chunks(text, settings.esi_source_path)

        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

        logger.info("creating_vector_store",
                   chunks_to_embed=len(documents),
                   embedding_model=settings.embedding_model)

        # Ensure the ChromaDB directory exists
        settings.chroma_db_path.mkdir(parents=True, exist_ok=True)

        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(settings.chroma_db_path),
        )

        # Persist the vector store
        vector_store.persist()

        # Test retrieval to verify ingestion worked
        test_query = "chest pain assessment"
        test_results = vector_store.similarity_search(test_query, k=2)

        logger.info("ingest_complete",
                    chunks=len(documents),
                    directory=str(settings.chroma_db_path),
                    source_file=str(settings.esi_source_path),
                    test_results=len(test_results))

        if test_results:
            logger.info("ingestion_verification",
                       query=test_query,
                       result_preview=test_results[0].page_content[:100] + "...")

        print(f"✅ Successfully ingested {len(documents)} chunks into ChromaDB")
        print(f"📁 Vector database stored at: {settings.chroma_db_path}")
        return 0

    except FileNotFoundError as e:
        logger.error("file_not_found", error=str(e))
        print(f"❌ File not found: {e}")
        return 1
    except ImportError as e:
        logger.error("missing_dependency", error=str(e))
        print(f"❌ Missing dependency: {e}")
        return 1
    except Exception as e:
        logger.error("ingestion_failed", error=str(e))
        print(f"❌ Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())