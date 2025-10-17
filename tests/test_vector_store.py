"""
Tests for vector store functionality
"""
import pytest
from unittest.mock import patch, Mock, MagicMock

from backend.services.vector_store import VectorStore


class TestVectorStore:
    """Test suite for VectorStore"""
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_settings):
        """Setup test environment"""
        with patch('backend.services.vector_store.get_settings', return_value=mock_settings):
            with patch('backend.services.vector_store.GoogleGenerativeAIEmbeddings'):
                with patch('backend.services.vector_store.chromadb.PersistentClient'):
                    with patch('backend.services.vector_store.Chroma'):
                        self.store = VectorStore()
    
    def test_initialization(self):
        """Test VectorStore initialization"""
        assert self.store is not None
        assert self.store.vector_store is not None
        assert self.store.embedding_function is not None
    
    def test_add_documents_success(self, sample_chunks):
        """Test successful document addition"""
        self.store.vector_store.add_texts = Mock(return_value=True)
        
        result = self.store.add_documents(sample_chunks)
        
        assert result is True
        self.store.vector_store.add_texts.assert_called_once()
        
        # Verify correct arguments
        call_args = self.store.vector_store.add_texts.call_args
        assert len(call_args.kwargs['texts']) == 3
        assert len(call_args.kwargs['metadatas']) == 3
    
    def test_add_documents_error(self, sample_chunks):
        """Test document addition with error"""
        self.store.vector_store.add_texts = Mock(
            side_effect=Exception("Database Error")
        )
        
        with pytest.raises(Exception) as exc_info:
            self.store.add_documents(sample_chunks)
        
        assert "Error adding documents" in str(exc_info.value)
    
    def test_similarity_search_success(self):
        """Test successful similarity search"""
        mock_doc = Mock()
        mock_doc.page_content = "Sample content"
        mock_doc.metadata = {"source": "test.pdf", "page": 0}
        
        self.store.vector_store.similarity_search_with_score = Mock(
            return_value=[(mock_doc, 0.95), (mock_doc, 0.85)]
        )
        
        results = self.store.similarity_search("test query", k=2)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["text"] == "Sample content"
        assert results[0]["metadata"]["source"] == "test.pdf"
        assert results[0]["score"] == 0.95
    
    def test_similarity_search_error(self):
        """Test similarity search with error"""
        self.store.vector_store.similarity_search_with_score = Mock(
            side_effect=Exception("Search Error")
        )
        
        with pytest.raises(Exception) as exc_info:
            self.store.similarity_search("test query")
        
        assert "Error performing similarity search" in str(exc_info.value)
    
    def test_get_retriever(self):
        """Test retriever creation"""
        self.store.vector_store.as_retriever = Mock(return_value=Mock())
        
        retriever = self.store.get_retriever()
        
        assert retriever is not None
        self.store.vector_store.as_retriever.assert_called_once()
    
    def test_get_collection_count_success(self):
        """Test getting collection count"""
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        self.store.client.get_collection = Mock(return_value=mock_collection)
        
        count = self.store.get_collection_count()
        
        assert count == 42
    
    def test_get_collection_count_error(self):
        """Test getting collection count with error"""
        self.store.client.get_collection = Mock(side_effect=Exception("Error"))
        
        count = self.store.get_collection_count()
        
        assert count == 0
    
    def test_clear_collection_success(self):
        """Test clearing collection"""
        self.store.client.delete_collection = Mock(return_value=True)
        
        with patch('backend.services.vector_store.Chroma'):
            result = self.store.clear_collection()
        
        assert result is True
        self.store.client.delete_collection.assert_called_once_with("documents")
