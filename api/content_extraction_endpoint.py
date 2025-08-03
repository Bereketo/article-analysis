from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

router = APIRouter(
    prefix="/api/content-extraction",
    tags=["content-extraction"],
    responses={404: {"description": "Not found"}},
)

# Keep app for backward compatibility but use router for routes
app = APIRouter()
