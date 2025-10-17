"""
Tests for document processing functionality
"""
import pytest
import os
from unittest.mock import patch, Mock
from pathlib import Path

from backend.services.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test suite for DocumentProcessor"""
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_settings):
        """Setup test environment"""
        with patch('backend.services.document_processor.get_settings', return_value=mock_settings):
            self.processor = DocumentProcessor()
    
    def test_initialization(self):
        """Test DocumentProcessor initialization"""
        assert self.processor is not None
        assert self.processor.text_splitter is not None
        assert self.processor.settings.CHUNK_SIZE == 500
        assert self.processor.settings.CHUNK_OVERLAP == 50
    
    @pytest.mark.asyncio
    async def test_process_pdf_success(self, sample_pdf_path, temp_test_dir, mock_settings):
        """Test successful PDF processing"""
        with patch('backend.services.document_processor.get_settings', return_value=mock_settings):
            processor = DocumentProcessor()
            
            # Process the PDF
            chunks = await processor.process_pdf(sample_pdf_path, "test_document.pdf")
            
            # Assertions
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            
            # Check chunk structure
            first_chunk = chunks[0]
            assert "text" in first_chunk
            assert "metadata" in first_chunk
            assert first_chunk["metadata"]["source"] == "test_document.pdf"
            assert "chunk_id" in first_chunk["metadata"]
    
    @pytest.mark.asyncio
    async def test_process_pdf_invalid_file(self, temp_test_dir, mock_settings):
        """Test processing with invalid file"""
        with patch('backend.services.document_processor.get_settings', return_value=mock_settings):
            processor = DocumentProcessor()
            
            invalid_path = os.path.join(temp_test_dir, "nonexistent.pdf")
            
            with pytest.raises(Exception):
                await processor.process_pdf(invalid_path, "nonexistent.pdf")
    
    def test_get_uploaded_files_empty(self, mock_settings):
        """Test getting uploaded files when none exist"""
        with patch('backend.services.document_processor.get_settings', return_value=mock_settings):
            processor = DocumentProcessor()
            files = processor.get_uploaded_files()
            assert isinstance(files, list)
    
    def test_get_uploaded_files_with_files(self, mock_settings, temp_test_dir):
        """Test getting uploaded files when files exist"""
        # Create test PDF files
        test_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        for filename in test_files:
            file_path = os.path.join(temp_test_dir, filename)
            Path(file_path).touch()
        
        mock_settings.UPLOAD_DIR = temp_test_dir
        
        with patch('backend.services.document_processor.get_settings', return_value=mock_settings):
            processor = DocumentProcessor()
            files = processor.get_uploaded_files()
            
            assert len(files) == 3
            assert all(f in files for f in test_files)
