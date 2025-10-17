from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import tempfile
import os

from backend.config import get_settings
from backend.services.document_processor import DocumentProcessor
from backend.services.vector_store import VectorStore
from backend.services.rag_chain import RAGChain
from backend.utils.helpers import validate_pdf, format_file_size, sanitize_filename


# Initialize FastAPI app
app = FastAPI(
    title="RAG Document QA API",
    description="API for document-based question answering using RAG",
    version="1.0.0"
)

# ✅ Add CORS middleware (your version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",                 # Local Streamlit frontend
        "https://rag-frontend.onrender.com",     # Render frontend
        "*"                                      # Allow all (for testing only)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
doc_processor = DocumentProcessor()
vector_store = VectorStore()
rag_chain = RAGChain()


# Pydantic models
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    question: str


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    documents_count: int
    uploaded_files: List[str]


# API Endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Document QA API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "query": "/query",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system stats"""
    try:
        doc_count = vector_store.get_collection_count()
        uploaded_files = doc_processor.get_uploaded_files()
        
        return HealthResponse(
            status="healthy",
            documents_count=doc_count,
            uploaded_files=uploaded_files
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document
    
    - **file**: PDF file to upload
    """
    try:
        # Validate file type
        if not validate_pdf(file.filename):  # type: ignore
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)  # type: ignore
        
        # Check file size
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Max size: {format_file_size(settings.MAX_UPLOAD_SIZE)}"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            chunks = await doc_processor.process_pdf(tmp_file_path, safe_filename)
            
            # Add to vector store
            vector_store.add_documents(chunks)
            
            return UploadResponse(
                filename=safe_filename,
                chunks_created=len(chunks),
                message=f"Successfully processed {safe_filename} into {len(chunks)} chunks"
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document collection
    
    - **question**: Question to ask about the documents
    """
    try:
        # Check if documents exist
        if vector_store.get_collection_count() == 0:
            raise HTTPException(
                status_code=400, 
                detail="No documents available. Please upload documents first."
            )
        
        # Process query
        result = rag_chain.query(request.question)
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store"""
    try:
        vector_store.clear_collection()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = rag_chain.get_store_stats()
        uploaded_files = doc_processor.get_uploaded_files()
        
        return {
            "vector_store": stats,
            "uploaded_files": {
                "count": len(uploaded_files),
                "files": uploaded_files
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
