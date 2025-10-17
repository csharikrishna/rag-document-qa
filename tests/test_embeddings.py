"""
Tests for embedding generation functionality
"""
import pytest
from unittest.mock import patch, Mock

from backend.services.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService"""
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_settings):
        """Setup test environment"""
        with patch('backend.services.embeddings.get_settings', return_value=mock_settings):
            with patch('backend.services.embeddings.GoogleGenerativeAIEmbeddings'):
                self.service = EmbeddingService()
    
    def test_initialization(self):
        """Test EmbeddingService initialization"""
        assert self.service is not None
        assert self.service.embedding_model is not None
    
    def test_embed_documents_success(self, mock_embedding_response):
        """Test successful document embedding generation"""
        # Mock the embedding model
        self.service.embedding_model.embed_documents = Mock(
            return_value=[mock_embedding_response, mock_embedding_response]
        )
        
        texts = ["First document text", "Second document text"]
        embeddings = self.service.embed_documents(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1000
        self.service.embedding_model.embed_documents.assert_called_once_with(texts)
    
    def test_embed_documents_error(self):
        """Test embedding generation with error"""
        self.service.embedding_model.embed_documents = Mock(
            side_effect=Exception("API Error")
        )
        
        texts = ["Test text"]
        
        with pytest.raises(Exception) as exc_info:
            self.service.embed_documents(texts)
        
        assert "Error generating embeddings" in str(exc_info.value)
    
    def test_embed_query_success(self, mock_embedding_response):
        """Test successful query embedding generation"""
        self.service.embedding_model.embed_query = Mock(
            return_value=mock_embedding_response
        )
        
        query = "What is this document about?"
        embedding = self.service.embed_query(query)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1000
        self.service.embedding_model.embed_query.assert_called_once_with(query)
    
    def test_embed_query_error(self):
        """Test query embedding with error"""
        self.service.embedding_model.embed_query = Mock(
            side_effect=Exception("API Error")
        )
        
        query = "Test query"
        
        with pytest.raises(Exception) as exc_info:
            self.service.embed_query(query)
        
        assert "Error generating query embedding" in str(exc_info.value)
    
    def test_embed_empty_list(self):
        """Test embedding empty document list"""
        self.service.embedding_model.embed_documents = Mock(return_value=[])
        
        embeddings = self.service.embed_documents([])
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 0
