from fastapi import FastAPI, HTTPException, status, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Any, Optional, Union
import logging
import re
from datetime import datetime
from urllib.parse import urlparse
from enum import Enum
import os
import json
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Content Extraction API",
    description="""
    Advanced content extraction API that uses AI to extract and clean content from web pages.
    Features:
    - Extracts main article content from any URL
    - Cleans and formats content for readability
    - Handles multiple URLs asynchronously
    - Uses AI for advanced content cleaning
    """,
    version="1.0.0"
)

# Configure logging
logger = logging.getLogger(__name__)

# Enums for content types
class ContentType(str, Enum):
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS_ARTICLE = "news_article"
    PRODUCT_PAGE = "product_page"
    FORUM = "forum"
    OTHER = "other"

# Pydantic models for request/response
class UrlExtractionRequest(BaseModel):
    """Individual URL extraction request"""
    url: HttpUrl = Field(..., description="URL of the page to extract content from")
    content_type: Optional[ContentType] = Field(
        ContentType.ARTICLE,
        description="Type of content expected at the URL"
    )
    include_metadata: Optional[bool] = Field(
        True,
        description="Whether to include metadata in the response"
    )
    include_raw_text: Optional[bool] = Field(
        False,
        description="Whether to include raw extracted text before cleaning"
    )

class ContentExtractionRequest(BaseModel):
    """Request model for content extraction"""
    urls: List[UrlExtractionRequest] = Field(
        ...,
        description="List of URLs to extract content from",
        min_items=1,
        max_items=50
    )
    aliases: List[str] = Field(
        ...,
        description="List of company name aliases to include in content analysis",
        min_items=1
    )
    parent_company_name: Optional[str] = Field(
        "Unknown",
        description="Name of the parent company for context"
    )
    language: Optional[str] = Field(
        "en",
        description="Language code for content processing (e.g., 'en', 'es', 'fr')",
        min_length=2,
        max_length=5
    )
    include_llm_cleaning: Optional[bool] = Field(
        True,
        description="Whether to use LLM for advanced content cleaning"
    )

class ContentMetadata(BaseModel):
    """Metadata about the extracted content"""
    url: HttpUrl = Field(..., description="Source URL of the content")
    title: Optional[str] = Field(None, description="Title of the web page")
    author: Optional[str] = Field(None, description="Author of the content")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    language: Optional[str] = Field(None, description="Detected language of the content")
    word_count: Optional[int] = Field(None, description="Number of words in the content")
    domain: Optional[str] = Field(None, description="Domain of the source URL")
    content_type: Optional[ContentType] = Field(None, description="Type of content")
    extraction_timestamp: datetime = Field(..., description="When the content was extracted")

class ExtractedContentItem(BaseModel):
    """Single extracted content item"""
    content: str = Field(..., description="The cleaned and formatted content")
    metadata: ContentMetadata = Field(..., description="Metadata about the content")
    raw_text: Optional[str] = Field(None, description="Raw extracted text before cleaning")
    processing_time_ms: Optional[float] = Field(None, description="Time taken to process in milliseconds")

class ProcessingSummary(BaseModel):
    """Summary of the content extraction process"""
    total_urls: int = Field(..., description="Total number of URLs processed")
    successful_extractions: int = Field(..., description="Number of successful extractions")
    failed_extractions: int = Field(..., description="Number of failed extractions")
    total_processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    average_processing_time_seconds: float = Field(..., description="Average processing time per URL in seconds")

class ContentExtractionResponse(BaseModel):
    """Response model for content extraction"""
    items: List[ExtractedContentItem] = Field(..., description="List of extracted content items")
    summary: ProcessingSummary = Field(..., description="Summary of the extraction process")
    request_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the request"
    )

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")

# Initialize the content extraction agent
extractor = ImprovedContentExtractionAgent()

def get_llm() -> AzureChatOpenAI:
    """Initialize and return the Azure OpenAI LLM"""
    return AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.1,
        max_tokens=4000
    )

@app.post(
    "/extract",
    response_model=ContentExtractionResponse,
    responses={
        200: {"description": "Content extracted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["content-extraction"],
    summary="Extract content from URLs",
    description="""
    Extract and clean content from one or more web pages.
    This endpoint processes multiple URLs asynchronously and returns the extracted content.
    """
)
async def extract_content(
    content_request: ContentExtractionRequest,
    x_api_key: Optional[str] = None
) -> ContentExtractionResponse:
    """
    Extract content from the provided URLs using advanced AI-powered extraction.
    
    - **urls**: List of URLs to extract content from
    - **aliases**: Company name aliases to include in content analysis
    - **parent_company_name**: (Optional) Parent company name
    - **language**: (Optional) Language code (default: en)
    - **include_llm_cleaning**: (Optional) Use LLM for advanced cleaning (default: true)
    
    Returns structured content with metadata and processing information.
    """
    try:
        logger.info(f"Starting content extraction for {len(content_request.urls)} URLs")
        
        # Initialize LLM if needed
        llm = get_llm() if content_request.include_llm_cleaning else None
        
        # Process each URL
        items = []
        for url_request in content_request.urls:
            try:
                # Extract content using the extraction agent
                result = await extractor.extract_content(url_request.url)
                
                # Format the content
                formatted_content = format_content(result.get("content", ""))
                
                # Apply LLM cleaning if enabled
                if content_request.include_llm_cleaning and llm:
                    formatted_content = await format_content_with_llm(formatted_content, llm)
                
                # Create metadata
                metadata = ContentMetadata(
                    url=url_request.url,
                    title=result.get("title"),
                    author=result.get("author"),
                    published_date=result.get("published_date"),
                    language=content_request.language,
                    word_count=len(formatted_content.split()),
                    domain=urlparse(str(url_request.url)).netloc,
                    content_type=url_request.content_type,
                    extraction_timestamp=datetime.utcnow()
                )
                
                # Create response item
                item = ExtractedContentItem(
                    content=formatted_content,
                    metadata=metadata,
                    raw_text=result.get("content") if url_request.include_raw_text else None,
                    processing_time_ms=result.get("processing_time_ms")
                )
                items.append(item)
                
            except Exception as e:
                logger.error(f"Error extracting content from {url_request.url}: {str(e)}")
                continue
        
        # Create processing summary
        summary = ProcessingSummary(
            total_urls=len(content_request.urls),
            successful_extractions=len(items),
            failed_extractions=len(content_request.urls) - len(items),
            total_processing_time_seconds=sum(
                (item.processing_time_ms or 0) / 1000 
                for item in items
            ),
            average_processing_time_seconds=(
                sum((item.processing_time_ms or 0) for item in items) / 
                (len(items) * 1000) if items else 0
            )
        )
        
        return ContentExtractionResponse(
            items=items,
            summary=summary,
            request_metadata={
                "aliases": content_request.aliases,
                "parent_company": content_request.parent_company_name,
                "language": content_request.language
            }
        )
        
    except Exception as e:
        logger.error(f"Content extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to extract content",
                "details": str(e)
            }
        )

@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the Content Extraction API service is running and healthy",
    responses={
        200: {"description": "API is healthy"},
        500: {"description": "API is not healthy"}
    }
)
async def health_check():
    """
    Health check endpoint that verifies the Content Extraction API service is running properly.
    
    Returns:
        dict: Status of the API and its components
    """
    try:
        # Add any additional health checks here
        return {
            "status": "healthy",
            "service": "content-extraction-api",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "extractor": "ready",
                "llm": "available" if os.getenv("AZURE_OPENAI_API_KEY") else "disabled"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.content_extraction_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )