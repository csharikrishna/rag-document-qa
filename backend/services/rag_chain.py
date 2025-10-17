from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Dict

from backend.config import get_settings
from backend.services.vector_store import VectorStore


class RAGChain:
    """Handles RAG query processing and answer generation"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.LLM_MODEL,
            google_api_key=self.settings.GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        # Initialize vector store
        self.vector_store = VectorStore()
        
        # Define prompt template
        self.prompt_template = """Using the following documents, answer the user's question succinctly and accurately.
If the answer cannot be found in the documents, say "I cannot find this information in the provided documents."

Documents:
{context}

Question: {question}

Answer:"""
        
        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> Dict:
        """
        Process a query and generate an answer
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        try:
            # Get retriever
            retriever = self.vector_store.get_retriever()
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.PROMPT}
            )
            
            # Run query
            result = qa_chain.invoke({"query": question})
            
            # Format response
            source_docs = []
            for doc in result.get("source_documents", []):
                source_docs.append({
                    "text": doc.page_content[:200] + "...",  # Truncate for display
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0)
                })
            
            return {
                "answer": result["result"],
                "sources": source_docs,
                "question": question
            }
            
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
    
    def get_store_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            "document_count": self.vector_store.get_collection_count()
        }
