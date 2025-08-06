from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import re
from datetime import datetime
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
import os

# Pydantic models for request/response
class ContentExtractionRequest(BaseModel):
    urls: List[str]
    aliases: List[str]
    parent_company_name: Optional[str] = "Unknown"

class SimplifiedExtractionData(BaseModel):
    urls: List[str]
    content: str
    aliases: List[str]
    parent_company_name: str

class ContentExtractionResponse(BaseModel):
    extracted_content: List[Dict[str, Any]]
    total_articles: int
    processing_summary: Dict[str, Any]
    simplified_data: SimplifiedExtractionData

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

async def _clean_content_with_llm(content: str, company_name: str) -> str:
    """Clean and format content using LLM to remove special characters and extract key info"""
    
    if not content or len(content.strip()) < 50:
        return content
    
    cleaning_prompt = f"""
    You are a content cleaning assistant. Clean the following article content by:
    
    1. Remove special characters, HTML tags, and formatting artifacts
    2. Fix broken sentences and paragraphs
    3. Remove navigation elements, ads, and irrelevant content
    4. Keep only the main article content related to "{company_name}"
    5. Maintain proper sentence structure and readability
    6. Remove duplicate sentences or paragraphs
    
    Original Content:
    {content[:3000]}  # Limit content length for LLM
    
    Return only the cleaned content without any explanations or metadata.
    """
    
    try:
        llm = AzureChatOpenAI(
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.1
        )
        
        response = await llm.ainvoke([HumanMessage(content=cleaning_prompt)])
        cleaned_content = response.content.strip()
        
        # Basic fallback cleaning if LLM fails
        if not cleaned_content or len(cleaned_content) < 20:
            cleaned_content = _basic_content_cleaning(content)
        
        return cleaned_content
        
    except Exception as e:
        logger.warning(f"LLM content cleaning failed: {e}")
        return _basic_content_cleaning(content)


def _basic_content_cleaning(content: str) -> str:
    """Basic content cleaning as fallback"""
    
    if not content:
        return ""
    
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Remove special characters but keep basic punctuation
    content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\%\$\&]', ' ', content)
    
    # Fix multiple spaces
    content = re.sub(r'\s+', ' ', content)
    
    # Remove very short lines (likely navigation/ads)
    lines = content.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
    
    return '\n'.join(cleaned_lines).strip()


router = APIRouter(
    prefix="/api/cdd",
    tags=["content-extraction"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post(
    "/extract",
    response_model=ContentExtractionResponse,
    responses={
        200: {"description": "Content extraction completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Extract content from URLs",
    description="""
    Extract and analyze content from a list of URLs using Jina AI and LLM analysis.
    """
)
async def extract_content(request: ContentExtractionRequest):
    try:
        logger.info(f"🔍 Starting content extraction for {len(request.urls)} URLs")
        
        # Validate URLs
        if not request.urls:
            raise HTTPException(
                status_code=422,
                detail="URLs list cannot be empty"
            )
        
        # Initialize the content extraction agent
        extractor = ImprovedContentExtractionAgent(
            num_results=10,
            concurrent_limit=24
        )
        
        # Process URLs directly using the agent's URL processing method
        search_results = []
        for url in request.urls:
            search_result = {
                "link": url,
                "title": "",
                "snippet": "",
                "source": url,
                "date": "",
                "source_query": "direct_url",
                "search_engine": "direct"
            }
            search_results.append(search_result)
        
        # Extract and analyze content
        processed_results = await extractor.extract_and_analyze_parallel(
            search_results, 
            request.aliases, 
            request.parent_company_name
        )
        
        # Format response to match the desired structure
        extracted_content = []
        for result in processed_results:
            jina_content = result.get("jina_content", {})
            raw_content = jina_content.get("content", "")

            # cleaned content
            cleaned_content = await _clean_content_with_llm(raw_content, request.parent_company_name)
            # Extract domain from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(result.get("link", ""))
            source_domain = parsed_url.netloc
            
            content_item = {
                "url": result.get("link", ""),
                "title": jina_content.get("title", result.get("title", "")),
                "content": cleaned_content,
                "metadata": {
                    "source_domain": source_domain,
                    "extracted_at": datetime.now().isoformat(),
                    "content_length": len(jina_content.get("content", "")),
                    "language": jina_content.get("language", "en"),
                    "extraction_status": result.get("extraction_status", "unknown"),
                    "extraction_metadata": {
                        "extractor": "jina-ai+llm",
                        "version": "2.0"
                    }
                }
            }
            extracted_content.append(content_item)
        
        # Create processing summary
        processing_summary = {
            "total_urls_requested": len(request.urls),
            "successful_extractions": len([r for r in processed_results if r.get("extraction_status") == "success"]),
            "failed_extractions": len([r for r in processed_results if r.get("extraction_status") != "success"]),
            "processing_time": datetime.now().isoformat(),
            "aliases_used": request.aliases,
            "parent_company": request.parent_company_name
        }
        
        response = ContentExtractionResponse(
            extracted_content=extracted_content,
            total_articles=len(extracted_content),
            processing_summary=processing_summary,
            simplified_data=SimplifiedExtractionData(
                urls=request.urls,
                content=cleaned_content,
                aliases=request.aliases,
                parent_company_name=request.parent_company_name
            )
        )
        
        logger.info(f"✅ Content extraction completed. Processed {len(extracted_content)} articles")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error during content extraction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract content: {str(e)}"
        )

@router.get("/content-extraction/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "content-extraction-api",
        "agent_type": "improved_content_extraction_agent"
    }
