"""
Tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import io

from backend.main import app


client = TestClient(app)


class TestAPI:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint_success(self):
        """Test health check endpoint"""
        with patch('backend.main.vector_store') as mock_store:
            with patch('backend.main.doc_processor') as mock_processor:
                mock_store.get_collection_count.return_value = 10
                mock_processor.get_uploaded_files.return_value = ["doc1.pdf", "doc2.pdf"]
                
                response = client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["documents_count"] == 10
                assert len(data["uploaded_files"]) == 2
    
    def test_upload_endpoint_success(self):
        """Test successful file upload"""
        # Create mock PDF file
        pdf_content = b"%PDF-1.4 test content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        
        with patch('backend.main.doc_processor') as mock_processor:
            with patch('backend.main.vector_store') as mock_store:
                mock_processor.process_pdf.return_value = [
                    {"text": "chunk1", "metadata": {}},
                    {"text": "chunk2", "metadata": {}}
                ]
                mock_store.add_documents.return_value = True
                
                response = client.post("/upload", files=files)
                
                assert response.status_code == 200
                data = response.json()
                assert "filename" in data
                assert data["chunks_created"] == 2
    
    def test_upload_endpoint_invalid_file_type(self):
        """Test upload with invalid file type"""
        files = {"file": ("test.txt", io.BytesIO(b"text content"), "text/plain")}
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 400
        assert "Only PDF files" in response.json()["detail"]
    
    def test_query_endpoint_success(self):
        """Test successful query"""
        with patch('backend.main.vector_store') as mock_store:
            with patch('backend.main.rag_chain') as mock_chain:
                mock_store.get_collection_count.return_value = 5
                mock_chain.query.return_value = {
                    "answer": "Test answer",
                    "sources": [],
                    "question": "Test question?"
                }
                
                response = client.post(
                    "/query",
                    json={"question": "Test question?"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["answer"] == "Test answer"
                assert "sources" in data
    
    def test_query_endpoint_no_documents(self):
        """Test query when no documents exist"""
        with patch('backend.main.vector_store') as mock_store:
            mock_store.get_collection_count.return_value = 0
            
            response = client.post(
                "/query",
                json={"question": "Test question?"}
            )
            
            assert response.status_code == 400
            assert "No documents available" in response.json()["detail"]
    
    def test_clear_documents_endpoint(self):
        """Test clearing documents"""
        with patch('backend.main.vector_store') as mock_store:
            mock_store.clear_collection.return_value = True
            
            response = client.delete("/documents")
            
            assert response.status_code == 200
            assert "cleared successfully" in response.json()["message"]
    
    def test_stats_endpoint(self):
        """Test stats endpoint"""
        with patch('backend.main.rag_chain') as mock_chain:
            with patch('backend.main.doc_processor') as mock_processor:
                mock_chain.get_store_stats.return_value = {"document_count": 15}
                mock_processor.get_uploaded_files.return_value = ["doc1.pdf"]
                
                response = client.get("/stats")
                
                assert response.status_code == 200
                data = response.json()
                assert "vector_store" in data
                assert "uploaded_files" in data
                assert data["uploaded_files"]["count"] == 1
