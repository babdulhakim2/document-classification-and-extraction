from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum

class DocumentCategory(str, Enum):
    INVOICE = "invoice"
    MARKETPLACE_SCREENSHOT = "marketplace_screenshot"
    CHAT_SCREENSHOT = "chat_screenshot"
    WEBSITE_SCREENSHOT = "website_screenshot"
    OTHER = "other"

class ExtractedEntities(BaseModel):
    class Config:
        extra = "allow"

class DocumentResult(BaseModel):
    category: DocumentCategory
    confidence: Optional[float] = None
    entities: ExtractedEntities
    raw_text: Optional[str] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
