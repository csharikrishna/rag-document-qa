"""
Configuration management for RAG Document QA
"""
import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    """Application configuration"""
    api_url: str
    max_file_size_mb: int
    request_timeout: int
    upload_timeout: int
    supported_file_types: list
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        return cls(
            api_url=os.getenv("API_URL", "http://localhost:8000"),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            upload_timeout=int(os.getenv("UPLOAD_TIMEOUT", "60")),
            supported_file_types=["pdf"]
        )

# Global config instance
config = Config.from_env()
