from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

# Add missing ErrorResponse model
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

app = FastAPI(
    title="Content Extraction API",
    description="Extract and process content from web pages.",
    version="1.0.0"
)
