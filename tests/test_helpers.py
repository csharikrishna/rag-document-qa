"""
Tests for utility helper functions
"""
import pytest
from backend.utils.helpers import (
    validate_pdf,
    format_file_size,
    sanitize_filename,
    truncate_text
)


class TestHelpers:
    """Test suite for helper functions"""
    
    def test_validate_pdf_valid(self):
        """Test PDF validation with valid filenames"""
        assert validate_pdf("document.pdf") is True
        assert validate_pdf("Document.PDF") is True
        assert validate_pdf("my_file.pdf") is True
    
    def test_validate_pdf_invalid(self):
        """Test PDF validation with invalid filenames"""
        assert validate_pdf("document.txt") is False
        assert validate_pdf("document.docx") is False
        assert validate_pdf("pdf") is False
    
    def test_format_file_size_bytes(self):
        """Test file size formatting for bytes"""
        assert "B" in format_file_size(512)
        assert "512" in format_file_size(512)
    
    def test_format_file_size_kilobytes(self):
        """Test file size formatting for kilobytes"""
        result = format_file_size(2048)
        assert "KB" in result
        assert "2.00" in result
    
    def test_format_file_size_megabytes(self):
        """Test file size formatting for megabytes"""
        result = format_file_size(5 * 1024 * 1024)
        assert "MB" in result
        assert "5.00" in result
    
    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization"""
        assert sanitize_filename("test.pdf") == "test.pdf"
        assert sanitize_filename("my_document.pdf") == "my_document.pdf"
    
    def test_sanitize_filename_with_path(self):
        """Test sanitizing filename with path traversal"""
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
    
    def test_sanitize_filename_special_chars(self):
        """Test sanitizing filename with special characters"""
        result = sanitize_filename("file@#$%.pdf")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
    
    def test_truncate_text_short(self):
        """Test truncating text shorter than max length"""
        text = "Short text"
        assert truncate_text(text, 100) == text
    
    def test_truncate_text_long(self):
        """Test truncating long text"""
        text = "A" * 200
        result = truncate_text(text, 100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")
    
    def test_truncate_text_exact(self):
        """Test truncating text at exact max length"""
        text = "A" * 100
        result = truncate_text(text, 100)
        assert result == text
        assert not result.endswith("...")
