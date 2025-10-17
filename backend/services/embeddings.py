from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List

from backend.config import get_settings


class EmbeddingService:
    """Handles text embedding generation using Gemini"""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=self.settings.EMBEDDING_MODEL,
            google_api_key=self.settings.GEMINI_API_KEY
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error generating query embedding: {str(e)}")
