from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import os
import asyncio
from datetime import datetime
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

# Pydantic models for request/response
class SerpRequest(BaseModel):
    search_queries: List[str]
    aliases: List[str]
    parent_company_name: Optional[str] = "Unknown"

class SerpResponse(BaseModel):
    results_data: Dict[str, Any]
    total_articles: int
    processing_summary: Dict[str, Any]

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

router = APIRouter(
    prefix="/api/serp",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post(
    "/search", 
    response_model=SerpResponse,
    responses={
        200: {"description": "Search completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Perform a web search",
    description="""
    Execute a search query across multiple search engines and content types.
    """
)
async def search_content(request: SerpRequest):
    try:
        logger.info(f"🔍 Starting content search with {len(request.search_queries)} queries")
        
        extractor = ImprovedContentExtractionAgent(
            num_results=10,
            concurrent_limit=24
        )
        
        search_results = extractor.extract_content(
            queries=request.search_queries,
            num_results=10
        )
        
        return search_results
        
    except Exception as e:
        logger.error(f"❌ Error during content search: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to perform content search: {str(e)}"
        )

@router.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "serp-content-search-api",
        "agent_type": "improved_content_extraction_agent"
    }
