"""
Main Streamlit application for RAG Document QA
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from api_client import APIClient
from config import config
from components import (
    render_upload_section,
    render_upload_stats,
    render_clear_documents,
    render_query_section
)

# Page configuration
st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main theme colors */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Improve button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    /* Improve expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }
    
    /* Improve metric styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4A4A4A;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-weight: 700;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border-color: #4A4A4A;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    """Get or create API client instance"""
    return APIClient(config.api_url, config.request_timeout)

def render_sidebar():
    """Render sidebar with system info and settings"""
    with st.sidebar:
        # Title with emoji
        st.markdown("# 📚 RAG Document QA")
        st.markdown("*Powered by AI*")
        st.divider()
        
        # Backend status
        st.markdown("### 🔌 System Status")
        api_client = get_api_client()
        health = api_client.check_health()
        
        if health["status"] == "connected":
            st.success("✅ Backend Connected")
            # Show backend info if available
            if "data" in health:
                data = health["data"]
                st.caption(f"Status: {data.get('status', 'running')}")
        else:
            st.error("❌ Backend Disconnected")
            st.warning(f"Error: {health.get('error', 'Unknown error')}")
            st.info("👉 Start the FastAPI backend:\n``````")
        
        st.divider()
        
        # Stats section
        st.markdown("### 📊 Statistics")
        render_upload_stats(api_client)
        
        st.divider()
        
        # Clear documents
        render_clear_documents(api_client)
        
        st.divider()
        
        # About section
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            **RAG Document QA System**
            
            Upload PDF documents and ask questions using AI-powered retrieval-augmented generation.
            
            **Tech Stack:**
            - 🤖 Gemini API for LLM
            - 🗄️ ChromaDB for vectors
            - 🔗 LangChain for RAG
            - ⚡ FastAPI backend
            - 🎨 Streamlit frontend
            
            **Features:**
            - PDF document upload
            - Intelligent Q&A
            - Source citation
            - Chat history
            - Export conversations
            """)
        
        # Footer
        st.divider()
        st.caption("Made with ❤️ using Streamlit")

def render_main_content():
    """Render main content area"""
    # Welcome header
    st.markdown("# 🤖 Document Q&A System")
    st.markdown("Upload your documents and ask questions to get AI-powered answers with source citations.")
    
    st.divider()
    
    # Get API client
    api_client = get_api_client()
    
    # Create tabs for upload and query
    tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])
    
    with tab1:
        render_upload_section(api_client)
        
        st.divider()
        
        # Tips section
        with st.expander("💡 Upload Tips"):
            st.markdown("""
            - **Supported Format:** PDF files only
            - **Maximum Size:** 10 MB per file
            - **Best Results:** Use clear, well-formatted documents
            - **Processing Time:** Depends on document size (typically 10-30 seconds)
            - **Multiple Files:** Upload files one at a time
            """)
    
    with tab2:
        # Check if documents are uploaded
        health = api_client.check_health()
        if health["status"] == "connected":
            doc_count = health.get("data", {}).get("documents_count", 0)
            if doc_count == 0:
                st.warning("⚠️ No documents uploaded yet. Please upload a PDF document first.")
                if st.button("➡️ Go to Upload Tab"):
                    st.switch_page
            else:
                render_query_section(api_client)
        else:
            st.error("❌ Cannot connect to backend. Please ensure the API server is running.")
        
        st.divider()
        
        # Query tips
        with st.expander("💡 Query Tips"):
            st.markdown("""
            - **Be Specific:** Ask clear, focused questions
            - **Use Context:** Reference specific topics from your documents
            - **Follow Up:** Ask related questions to dive deeper
            - **Check Sources:** Review source citations for accuracy
            - **Export Chat:** Save your conversation for later reference
            """)

def main():
    """Main application entry point"""
    # Initialize session state
    if 'confirm_clear' not in st.session_state:
        st.session_state['confirm_clear'] = False
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_main_content()

if __name__ == "__main__":
    main()
