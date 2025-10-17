"""
Upload component for document management
"""
import streamlit as st
from typing import Optional, Dict
from api_client import APIClient
from constants import MESSAGES, MAX_FILE_SIZE_BYTES

def validate_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded file"""
    if uploaded_file is None:
        return False, "No file selected"
    
    # Check file size
    file_size = uploaded_file.size
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return False, f"File too large ({size_mb:.1f}MB). Maximum size is 10MB."
    
    # Check file type
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    return True, ""

def render_upload_section(api_client: APIClient) -> Optional[Dict]:
    """
    Render the document upload section with validation
    
    Args:
        api_client: API client instance
        
    Returns:
        Upload response or None
    """
    st.markdown("### 📄 Upload Documents")
    st.markdown("Upload PDF documents to build your knowledge base.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Select a PDF file to upload (max 10MB)",
        key="file_uploader"
    )
    
    # Show file info if selected
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📎 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
    
    # Upload button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        upload_button = st.button(
            "🚀 Upload & Process",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True
        )
    
    if upload_button and uploaded_file is not None:
        # Validate file
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return None
        
        # Upload file
        with st.spinner(MESSAGES["processing"]):
            result = api_client.upload_document(
                uploaded_file.name,
                uploaded_file.getvalue(),
                timeout=60
            )
            
            if result:
                st.success(MESSAGES["upload_success"])
                
                # Show details in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Filename", result.get('filename', 'N/A'))
                with col2:
                    st.metric("📊 Chunks Created", result.get('chunks_created', 0))
                
                # Show detailed info in expander
                with st.expander("🔍 Processing Details"):
                    st.json(result)
                
                return result
    
    return None

def render_upload_stats(api_client: APIClient):
    """
    Render upload statistics and system info
    
    Args:
        api_client: API client instance
    """
    stats = api_client.get_stats()
    
    if stats:
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            doc_count = stats.get("vector_store", {}).get("document_count", 0)
            st.metric(
                "📚 Total Chunks",
                doc_count,
                help="Total number of document chunks in vector store"
            )
        
        with col2:
            file_count = stats.get("uploaded_files", {}).get("count", 0)
            st.metric(
                "📁 Uploaded Files",
                file_count,
                help="Number of files uploaded"
            )
        
        with col3:
            # Calculate average chunks per file
            avg_chunks = doc_count // file_count if file_count > 0 else 0
            st.metric(
                "⚡ Avg Chunks/File",
                avg_chunks,
                help="Average chunks per document"
            )
        
        # Show file list if available
        files = stats.get("uploaded_files", {}).get("files", [])
        if files:
            with st.expander(f"📋 View {len(files)} Uploaded File(s)"):
                for idx, file in enumerate(files, 1):
                    st.markdown(f"{idx}. 📄 **{file}**")
    else:
        st.info(MESSAGES["no_documents"])

def render_clear_documents(api_client: APIClient):
    """
    Render clear documents button with confirmation
    
    Args:
        api_client: API client instance
    """
    st.divider()
    
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.warning("⚠️ This will permanently delete all uploaded documents from the vector store.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                # Use session state for confirmation
                st.session_state['confirm_clear'] = True
        
        # Show confirmation dialog
        if st.session_state.get('confirm_clear', False):
            st.error("⚠️ Are you sure? This action cannot be undone!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete All", type="primary", use_container_width=True):
                    if api_client.clear_documents():
                        st.success("✅ All documents cleared successfully!")
                        st.session_state['confirm_clear'] = False
                        st.session_state['messages'] = []  # Clear chat history
                        st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state['confirm_clear'] = False
                    st.rerun()
