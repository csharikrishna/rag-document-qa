"""
Pytest configuration and shared fixtures
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    return Settings(
        GEMINI_API_KEY="test_api_key",
        CHROMA_PERSIST_DIR="./test_data/chroma_db",
        UPLOAD_DIR="./test_data/uploads",
        EMBEDDING_MODEL="models/text-embedding-004",
        LLM_MODEL="gemini-2.0-flash-exp",
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=50,
        TOP_K_RESULTS=3
    )


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test files"""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def sample_pdf_path(temp_test_dir):
    """Create a sample PDF file for testing"""
    from reportlab.pdfgen import canvas
    
    pdf_path = os.path.join(temp_test_dir, "test_document.pdf")
    
    # Create a simple PDF
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 750, "This is a test document.")
    c.drawString(100, 730, "It contains sample text for testing.")
    c.drawString(100, 710, "The RAG system should process this document.")
    c.showPage()
    c.save()
    
    return pdf_path


@pytest.fixture
def mock_embedding_response():
    """Mock embedding vector response"""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 200  # 1000-dim vector


@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    return {
        "result": "This is a test answer from the RAG system.",
        "source_documents": [
            Mock(
                page_content="Sample document content",
                metadata={"source": "test.pdf", "page": 0}
            )
        ]
    }


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing"""
    return [
        {
            "text": "This is the first chunk of text from the document.",
            "metadata": {"source": "test.pdf", "page": 0, "chunk_id": 0}
        },
        {
            "text": "This is the second chunk of text from the document.",
            "metadata": {"source": "test.pdf", "page": 0, "chunk_id": 1}
        },
        {
            "text": "This is the third chunk of text from the document.",
            "metadata": {"source": "test.pdf", "page": 1, "chunk_id": 2}
        }
    ]


@pytest.fixture
def cleanup_test_data():
    """Cleanup test data after tests"""
    yield
    # Cleanup test directories
    test_dirs = ["./test_data", "./data/test"]
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
