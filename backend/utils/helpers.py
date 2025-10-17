import os
from typing import Optional


def validate_pdf(filename: str) -> bool:
    """
    Validate if file is a PDF
    
    Args:
        filename: Name of the file
        
    Returns:
        True if valid PDF, False otherwise
    """
    return filename.lower().endswith('.pdf')


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0 # type: ignore
    return f"{size_bytes:.2f} TB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Get just the filename, remove path components
    filename = os.path.basename(filename)
    
    # Remove any non-alphanumeric characters except dots, dashes, underscores
    import re
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    
    return filename


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
