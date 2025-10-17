from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Optional
import chromadb
from pathlib import Path

from backend.config import get_settings


class VectorStore:
    """Manages ChromaDB vector storage operations"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Create persist directory
        Path(self.settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding function
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=self.settings.EMBEDDING_MODEL,
            google_api_key=self.settings.GEMINI_API_KEY
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.settings.CHROMA_PERSIST_DIR
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.client,
            collection_name="documents",
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, chunks: List[dict]) -> bool:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of document chunks with text and metadata
            
        Returns:
            Success status
        """
        try:
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            if k is None:
                k = self.settings.TOP_K_RESULTS
            
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")
    
    def get_retriever(self):
        """Get LangChain retriever for RAG chain"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": self.settings.TOP_K_RESULTS}
        )
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        try:
            collection = self.client.get_collection("documents")
            return collection.count()
        except:
            return 0
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            self.client.delete_collection("documents")
            self.vector_store = Chroma(
                client=self.client,
                collection_name="documents",
                embedding_function=self.embedding_function
            )
            return True
        except Exception as e:
            raise Exception(f"Error clearing collection: {str(e)}")
