import os
from typing import Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment once
load_dotenv()

# Constants
SUPPORTED_FILE_TYPES = [
    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp",
    "application/pdf",
    "text/plain", "text/csv",
    "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
]

AVAILABLE_MODELS = {
    "gemini-flash-2.5": "google/gemini-2.5-flash",
    "gpt-4o": "openai/gpt-4o"
}

IMAGE_SIZE_THRESHOLDS = {
    "large_image_pixels": 2000 * 2000,  # 4MP
    "thumbnail_size": (1600, 1600),
    "jpeg_threshold_pixels": 1000 * 1000,  # 1MP
}

class Settings(BaseModel):
    """Simple application settings"""
    
    # API
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    available_models: Dict[str, str] = AVAILABLE_MODELS
    default_model: str = "gpt-4o" 
    judge_model: str = "gpt-4o"
    
    # Processing
    max_file_size_mb: int = 10
    max_files_per_batch: int = 20
    max_tokens: int = 4082
    temperature: float = 0.0
    
    # PDF Processing
    pdf_engine: str = "mistral-ocr"  # "pdf-text" (free) or "mistral-ocr" (paid)
    
    # Performance  
    request_timeout: int = 120
    max_concurrent_requests: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # UI
    log_level: str = "INFO"
    page_title: str = "Document Classifier"
    page_icon: str = "📄"

# Load settings from environment
settings = Settings(
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
    max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
    max_files_per_batch=int(os.getenv("MAX_FILES_PER_BATCH", "20")),
    request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
    max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
    max_retries=int(os.getenv("MAX_RETRIES", "3")),
    retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
    max_tokens=int(os.getenv("MAX_TOKENS", "4082")),
    temperature=float(os.getenv("TEMPERATURE", "0.0")),
    judge_model=os.getenv("JUDGE_MODEL", "gpt-4o"),
    default_model=os.getenv("DEFAULT_MODEL", "gemini-flash-2.5"),
    pdf_engine=os.getenv("PDF_ENGINE", "pdf-text"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    page_title=os.getenv("PAGE_TITLE", "Document Classifier"),
    page_icon=os.getenv("PAGE_ICON", "📄")
)

# Simple validation functions
def validate_file_size(file_size_bytes: int) -> bool:
    """Check if file size is within limits"""
    return file_size_bytes <= settings.max_file_size_mb * 1024 * 1024

def validate_file_type(mime_type: str) -> bool:
    """Check if file type is supported"""
    return mime_type in SUPPORTED_FILE_TYPES

def get_request_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Get API request headers"""
    return {
        "Authorization": f"Bearer {api_key or settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "X-Title": "Document Classifier"
    }

# Helper: derive allowed file extensions from supported MIME types
def get_supported_extensions() -> list[str]:
    """Return a de-duplicated list of file extensions allowed by SUPPORTED_FILE_TYPES."""
    mime_to_exts = {
        "image/png": ["png"],
        "image/jpeg": ["jpg", "jpeg"],
        "image/jpg": ["jpg", "jpeg"],
        "image/gif": ["gif"],
        "image/bmp": ["bmp"],
        "application/pdf": ["pdf"],
        "text/plain": ["txt"],
        "text/csv": ["csv"],
        "application/msword": ["doc"],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ["docx"],
    }
    exts: set[str] = set()
    for mime in SUPPORTED_FILE_TYPES:
        for ext in mime_to_exts.get(mime, []):
            exts.add(ext)
    # Return stable ordering
    priority = ["pdf","png","jpg","jpeg","gif","bmp","txt","csv","doc","docx"]
    ordered = [e for e in priority if e in exts]
    # Add any others not in priority just in case
    ordered += sorted(exts - set(ordered))
    return ordered
