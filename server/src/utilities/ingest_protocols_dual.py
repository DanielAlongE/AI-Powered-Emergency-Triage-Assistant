#!/usr/bin/env python3
"""
Dual ingestion script to create both OpenAI and Ollama embedding databases.
Ingests ESI protocol content into both ChromaDB instances for maximum flexibility.
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
from services.ollama_embeddings import get_ollama_embedding_service, is_ollama_embeddings_available

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
    # Improved chunk configuration for better RAG performance
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )

    documents = splitter.create_documents(
        [text],
        metadatas=[{"source": str(source_path)}]
    )

    logger.info("text_split",
                total_chunks=len(documents),
                chunk_size=1000,
                chunk_overlap=200)

    # Log first few chunks for debugging
    for i, doc in enumerate(documents[:3]):
        logger.info(f"chunk_{i}",
                   length=len(doc.page_content),
                   preview=doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content)

    return documents


def ingest_openai_embeddings(documents: List[Document], settings) -> bool:
    """
    Ingest documents to OpenAI embeddings ChromaDB.

    Returns:
        True if successful, False otherwise
    """
    if not settings.openai_api_key:
        logger.warning("openai_api_key_missing",
                      message="Skipping OpenAI embeddings ingestion - no API key")
        return False

    try:
        logger.info("starting_openai_ingestion",
                   chunks=len(documents),
                   directory=str(settings.chroma_openai_path))

        # Create OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

        # Ensure the ChromaDB directory exists
        settings.chroma_openai_path.mkdir(parents=True, exist_ok=True)

        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(settings.chroma_openai_path),
        )

        # Persist the vector store
        vector_store.persist()

        # Test retrieval
        test_results = vector_store.similarity_search("chest pain assessment", k=2)

        logger.info("openai_ingestion_complete",
                   chunks=len(documents),
                   directory=str(settings.chroma_openai_path),
                   test_results=len(test_results),
                   embedding_model=settings.embedding_model)

        print(f"✅ OpenAI embeddings: Successfully ingested {len(documents)} chunks")
        print(f"📁 Stored at: {settings.chroma_openai_path}")
        return True

    except Exception as e:
        logger.error("openai_ingestion_failed", error=str(e))
        print(f"❌ OpenAI embeddings ingestion failed: {e}")
        return False


def ingest_ollama_embeddings(documents: List[Document], settings) -> bool:
    """
    Ingest documents to Ollama embeddings ChromaDB.

    Returns:
        True if successful, False otherwise
    """
    if not is_ollama_embeddings_available():
        logger.warning("ollama_embeddings_unavailable",
                      message="Skipping Ollama embeddings ingestion - service unavailable")
        print(f"⚠️  Ollama embeddings unavailable. Make sure:")
        print(f"   1. Ollama is running: ollama serve")
        print(f"   2. Model is available: ollama pull {settings.ollama_embedding_model}")
        return False

    try:
        logger.info("starting_ollama_ingestion",
                   chunks=len(documents),
                   directory=str(settings.chroma_ollama_path))

        # Get Ollama embedding service
        embedding_service = get_ollama_embedding_service()

        # Ensure the ChromaDB directory exists
        settings.chroma_ollama_path.mkdir(parents=True, exist_ok=True)

        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_service.embeddings,
            persist_directory=str(settings.chroma_ollama_path),
        )

        # Persist the vector store
        vector_store.persist()

        # Test retrieval
        test_results = vector_store.similarity_search("chest pain assessment", k=2)

        logger.info("ollama_ingestion_complete",
                   chunks=len(documents),
                   directory=str(settings.chroma_ollama_path),
                   test_results=len(test_results),
                   embedding_model=settings.ollama_embedding_model)

        print(f"✅ Ollama embeddings: Successfully ingested {len(documents)} chunks")
        print(f"📁 Stored at: {settings.chroma_ollama_path}")
        return True

    except Exception as e:
        logger.error("ollama_ingestion_failed", error=str(e))
        print(f"❌ Ollama embeddings ingestion failed: {e}")
        return False


def main():
    """
    Main dual ingestion function - creates both embedding databases.
    """
    settings = get_settings()

    # Configure logging
    configure_logging(settings.log_level)

    logger.info("dual_ingest_start",
                source_path=str(settings.esi_source_path),
                openai_path=str(settings.chroma_openai_path),
                ollama_path=str(settings.chroma_ollama_path))

    print("🚀 Starting dual embeddings ingestion...")
    print(f"📖 Source: {settings.esi_source_path}")
    print(f"🔍 OpenAI model: {settings.embedding_model}")
    print(f"🔍 Ollama model: {settings.ollama_embedding_model}")
    print()

    try:
        # Load and chunk the source text
        text = load_protocol_text(settings.esi_source_path)
        documents = create_document_chunks(text, settings.esi_source_path)

        logger.info("text_processed",
                    source=str(settings.esi_source_path),
                    text_length=len(text),
                    chunks=len(documents))

        print(f"📝 Processed {len(documents)} document chunks")
        print()

        # Track ingestion results
        openai_success = False
        ollama_success = False

        # Ingest to OpenAI embeddings
        print("🔄 Ingesting to OpenAI embeddings...")
        openai_success = ingest_openai_embeddings(documents, settings)

        print()

        # Ingest to Ollama embeddings
        print("🔄 Ingesting to Ollama embeddings...")
        ollama_success = ingest_ollama_embeddings(documents, settings)

        print()

        # Summary
        print("📊 Ingestion Summary:")
        print(f"   OpenAI embeddings: {'✅ Success' if openai_success else '❌ Failed'}")
        print(f"   Ollama embeddings: {'✅ Success' if ollama_success else '❌ Failed'}")

        if openai_success or ollama_success:
            print()
            print("🎉 Dual ingestion completed successfully!")
            if openai_success and ollama_success:
                print("   Both embedding types are now available.")
            elif openai_success:
                print("   OpenAI embeddings available. Set up Ollama for the second type.")
            else:
                print("   Ollama embeddings available. Add OpenAI API key for the second type.")

            return 0
        else:
            print()
            print("💥 Dual ingestion failed - no embeddings were created.")
            print("   Check the logs above for specific error details.")
            return 1

    except FileNotFoundError as e:
        logger.error("file_not_found", error=str(e))
        print(f"❌ File not found: {e}")
        return 1
    except ImportError as e:
        logger.error("missing_dependency", error=str(e))
        print(f"❌ Missing dependency: {e}")
        return 1
    except Exception as e:
        logger.error("dual_ingestion_failed", error=str(e))
        print(f"❌ Dual ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)