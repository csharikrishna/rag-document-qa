"""
Application constants
"""

# UI Messages
MESSAGES = {
    "upload_success": "✅ Document uploaded and processed successfully!",
    "upload_failed": "❌ Failed to upload document",
    "backend_disconnected": "❌ Backend is disconnected. Please start the FastAPI server.",
    "processing": "⏳ Processing your document...",
    "thinking": "🤔 Thinking...",
    "no_documents": "ℹ️ No documents uploaded yet. Upload a PDF to get started!",
    "chat_cleared": "🗑️ Chat history cleared",
}

# UI Colors (for custom styling)
COLORS = {
    "primary": "#FF4B4B",
    "secondary": "#0068C9",
    "success": "#00C851",
    "warning": "#FF8800",
    "error": "#FF4444",
    "background": "#0E1117",
    "card": "#262730",
}

# File constraints
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_FILE_TYPES = ["pdf"]
