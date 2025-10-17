"""
Tests for RAG chain functionality
"""
import pytest
from unittest.mock import patch, Mock

from backend.services.rag_chain import RAGChain


class TestRAGChain:
    """Test suite for RAGChain"""
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_settings):
        """Setup test environment"""
        with patch('backend.services.rag_chain.get_settings', return_value=mock_settings):
            with patch('backend.services.rag_chain.ChatGoogleGenerativeAI'):
                with patch('backend.services.rag_chain.VectorStore'):
                    self.chain = RAGChain()
    
    def test_initialization(self):
        """Test RAGChain initialization"""
        assert self.chain is not None
        assert self.chain.llm is not None
        assert self.chain.vector_store is not None
        assert self.chain.prompt_template is not None
    
    def test_query_success(self, mock_llm_response):
        """Test successful query processing"""
        # Mock retriever
        mock_retriever = Mock()
        self.chain.vector_store.get_retriever = Mock(return_value=mock_retriever)
        
        # Mock QA chain
        with patch('backend.services.rag_chain.RetrievalQA') as mock_qa:
            mock_qa_instance = Mock()
            mock_qa_instance.invoke = Mock(return_value=mock_llm_response)
            mock_qa.from_chain_type = Mock(return_value=mock_qa_instance)
            
            result = self.chain.query("What is this about?")
            
            assert isinstance(result, dict)
            assert "answer" in result
            assert "sources" in result
            assert "question" in result
            assert result["answer"] == mock_llm_response["result"]
            assert result["question"] == "What is this about?"
    
    def test_query_error(self):
        """Test query with error"""
        self.chain.vector_store.get_retriever = Mock(
            side_effect=Exception("Retrieval Error")
        )
        
        with pytest.raises(Exception) as exc_info:
            self.chain.query("Test question")
        
        assert "Error processing query" in str(exc_info.value)
    
    def test_query_empty_question(self, mock_llm_response):
        """Test query with empty question"""
        mock_retriever = Mock()
        self.chain.vector_store.get_retriever = Mock(return_value=mock_retriever)
        
        with patch('backend.services.rag_chain.RetrievalQA') as mock_qa:
            mock_qa_instance = Mock()
            mock_qa_instance.invoke = Mock(return_value=mock_llm_response)
            mock_qa.from_chain_type = Mock(return_value=mock_qa_instance)
            
            result = self.chain.query("")
            
            assert isinstance(result, dict)
            assert result["question"] == ""
    
    def test_get_store_stats(self):
        """Test getting vector store statistics"""
        self.chain.vector_store.get_collection_count = Mock(return_value=25)
        
        stats = self.chain.get_store_stats()
        
        assert isinstance(stats, dict)
        assert "document_count" in stats
        assert stats["document_count"] == 25
    
    def test_source_formatting(self):
        """Test that sources are properly formatted"""
        mock_doc = Mock()
        mock_doc.page_content = "A" * 300  # Long content
        mock_doc.metadata = {"source": "test.pdf", "page": 5}
        
        mock_response = {
            "result": "Test answer",
            "source_documents": [mock_doc]
        }
        
        mock_retriever = Mock()
        self.chain.vector_store.get_retriever = Mock(return_value=mock_retriever)
        
        with patch('backend.services.rag_chain.RetrievalQA') as mock_qa:
            mock_qa_instance = Mock()
            mock_qa_instance.invoke = Mock(return_value=mock_response)
            mock_qa.from_chain_type = Mock(return_value=mock_qa_instance)
            
            result = self.chain.query("Test?")
            
            # Check that text is truncated
            assert len(result["sources"][0]["text"]) <= 203  # 200 + "..."
            assert result["sources"][0]["text"].endswith("...")
            assert result["sources"][0]["page"] == 5
