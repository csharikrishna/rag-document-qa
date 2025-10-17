from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import os
import shutil
from pathlib import Path

from backend.config import get_settings


class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create upload directory if it doesn't exist
        Path(self.settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    
    async def process_pdf(self, file_path: str, filename: str) -> List[dict]:
        """
        Load and process a PDF file into chunks
        
        Args:
            file_path: Temporary file path
            filename: Original filename
            
        Returns:
            List of document chunks with metadata
        """
        try:
            # Save file to upload directory
            destination = os.path.join(self.settings.UPLOAD_DIR, filename)
            shutil.copy(file_path, destination)
            
            # Load PDF
            loader = PyPDFLoader(destination)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Format chunks with metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    "text": chunk.page_content,
                    "metadata": {
                        "source": filename,
                        "page": chunk.metadata.get("page", 0),
                        "chunk_id": i
                    }
                })
            
            return processed_chunks
            
        except Exception as e:
            raise Exception(f"Error processing PDF {filename}: {str(e)}")
    
    def get_uploaded_files(self) -> List[str]:
        """Get list of uploaded file names"""
        upload_dir = Path(self.settings.UPLOAD_DIR)
        if not upload_dir.exists():
            return []
        return [f.name for f in upload_dir.glob("*.pdf")]
