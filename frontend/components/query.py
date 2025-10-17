"""
Query component for document Q&A
"""
import streamlit as st
from typing import Optional, Dict, List
from api_client import APIClient
from constants import MESSAGES

def render_query_section(api_client: APIClient):
    """
    Render the query interface with chat-like experience
    
    Args:
        api_client: API client instance
    """
    st.markdown("### 💬 Ask Questions")
    st.markdown("Ask questions about your uploaded documents and get AI-powered answers.")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                render_sources(message["sources"])
    
    # Query input
    if question := st.chat_input("Type your question here...", key="chat_input"):
        # Validate input
        if len(question.strip()) == 0:
            st.warning("⚠️ Please enter a valid question")
            return
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer from backend
        with st.chat_message("assistant"):
            with st.spinner(MESSAGES["thinking"]):
                response_data = api_client.query_documents(question)
                
                if response_data:
                    answer = response_data.get("answer", "No answer received")
                    sources = response_data.get("sources", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        render_sources(sources)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = "Sorry, I couldn't process your question. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Chat controls at the bottom
    if st.session_state.messages:
        st.divider()
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col3:
            if st.button("💾 Export Chat", use_container_width=True):
                export_chat_history()

def render_sources(sources: List[Dict]):
    """
    Render source documents in an expandable section
    
    Args:
        sources: List of source documents
    """
    if sources and len(sources) > 0:
        with st.expander(f"📚 View {len(sources)} Source(s)", expanded=False):
            for idx, source in enumerate(sources, 1):
                # Create a card-like display for each source
                st.markdown(f"**📄 Source {idx}**")
                
                # Display source metadata
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.caption(f"📁 File: {source.get('source', 'Unknown')}")
                with col2:
                    st.caption(f"📄 Page: {source.get('page', 'N/A')}")
                
                # Display source text
                st.markdown(f"_{source.get('text', 'No text available')}_")
                
                if idx < len(sources):
                    st.divider()

def export_chat_history():
    """Export chat history as markdown"""
    if not st.session_state.messages:
        st.warning("No chat history to export")
        return
    
    # Generate markdown content
    markdown_content = "# Chat History\n\n"
    markdown_content += "---\n\n"
    
    for message in st.session_state.messages:
        role = "**You**" if message["role"] == "user" else "**Assistant**"
        markdown_content += f"{role}:\n{message['content']}\n\n"
        
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            markdown_content += "**Sources:**\n"
            for idx, source in enumerate(message["sources"], 1):
                markdown_content += f"{idx}. {source.get('source', 'Unknown')} (Page {source.get('page', 'N/A')})\n"
            markdown_content += "\n"
        
        markdown_content += "---\n\n"
    
    # Provide download button
    st.download_button(
        label="📥 Download Chat History",
        data=markdown_content,
        file_name="chat_history.md",
        mime="text/markdown",
        use_container_width=True
    )
    st.success("✅ Chat history ready for download!")
