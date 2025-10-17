"""
API client for backend communication
"""
import requests
import streamlit as st
from typing import Optional, Dict, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)

class APIClient:
    """Centralized API client with retry logic"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def check_health(self) -> Dict:
        """Check backend health status"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return {"status": "connected", "data": response.json()}
        except requests.exceptions.ConnectionError:
            return {"status": "disconnected", "error": "Cannot connect to backend"}
        except requests.exceptions.Timeout:
            return {"status": "timeout", "error": "Backend not responding"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def upload_document(self, file_name: str, file_content: bytes, timeout: int = 60) -> Optional[Dict]:
        """Upload document to backend"""
        try:
            files = {"file": (file_name, file_content, "application/pdf")}
            response = self.session.post(
                f"{self.base_url}/upload",
                files=files,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"❌ Upload failed: {error_detail}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("❌ Upload timed out. Please try with a smaller file.")
            return None
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Please ensure the API server is running.")
            return None
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            st.error(f"❌ Upload error: {str(e)}")
            return None
    
    def query_documents(self, question: str) -> Optional[Dict]:
        """Query documents with a question"""
        try:
            response = self.session.post(
                f"{self.base_url}/query",
                json={"question": question},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"❌ Query failed: {error_detail}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Please try again.")
            return None
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Please ensure the API server is running.")
            return None
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            st.error(f"❌ Query error: {str(e)}")
            return None
    
    @st.cache_data(ttl=10)
    def get_stats(_self) -> Optional[Dict]:
        """Get upload statistics (cached for 10 seconds)"""
        try:
            response = _self.session.get(
                f"{_self.base_url}/stats",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Stats fetch failed: {str(e)}")
            return None
    
    def clear_documents(self) -> bool:
        """Clear all documents from vector store"""
        try:
            response = self.session.delete(
                f"{self.base_url}/documents",
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Clear documents failed: {str(e)}")
            st.error(f"❌ Failed to clear documents: {str(e)}")
            return False
