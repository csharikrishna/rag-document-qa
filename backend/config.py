from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Keys
    GEMINI_API_KEY: str

    # Database paths
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    UPLOAD_DIR: str = "./data/uploaded_docs"
    
    # Model settings
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    LLM_MODEL: str = "gemini-2.0-flash-exp"
    
    # Chunking settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval settings
    TOP_K_RESULTS: int = 5
    
    # API settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

    # --- Added to fix Render validation errors ---
    API_URL: str = "http://localhost:8000"
    MAX_FILE_SIZE_MB: int = 10
    REQUEST_TIMEOUT: int = 30
    UPLOAD_TIMEOUT: int = 60
    # ------------------------------------------------

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # ensures extra env vars won’t crash the app


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
