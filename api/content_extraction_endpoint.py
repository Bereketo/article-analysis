from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import re
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

# Simple database tracking (optional)
try:
    from services.database_service import SimpleReportDB
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    SimpleReportDB = None

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
    Extract content from a list of URLs using Jina AI. This endpoint only extracts content without performing analysis.
    For content analysis, use the /api/cdd/article-analysis endpoint.
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
        
        # Extract content only (no analysis)
        processed_results = await extractor.extract_content_only_parallel(search_results)
        
        # Format response to match the desired structure
        extracted_content = []
        for result in processed_results:
            jina_content = result.get("jina_content", {})
            raw_content = jina_content.get("content", "")

            # Use basic content cleaning instead of LLM (much faster)
            cleaned_content = _basic_content_cleaning(raw_content)
            
            # Extract domain from URL
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
                        "extractor": "jina-ai+basic-cleaning",
                        "version": "2.1"
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
                content="".join([item.get('content', '') for item in extracted_content[:5]]),  # Sample of first 5 articles
                aliases=request.aliases,
                parent_company_name=request.parent_company_name
            )
        )
        
        # Save extraction results to JSON file in extracted-content directory
        try:
            
            # Create extracted-content directory if it doesn't exist
            output_dir = "extracted-content"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            # Clean company name for filename
            clean_company_name = re.sub(r'[^\w\s-]', '', request.parent_company_name).replace(' ', '_')
            filename = f"{clean_company_name}_content-extraction_{timestamp_str}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Prepare data structure for JSON file (similar to your existing format)
            json_data = {
                "metadata": {
                    "company_name": request.parent_company_name,
                    "aliases": request.aliases,
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_urls_requested": len(request.urls),
                    "total_urls_processed": len(processed_results),
                    "extraction_statistics": {
                        "successful": processing_summary["successful_extractions"],
                        "failed": processing_summary["failed_extractions"],
                        "success_rate": processing_summary["successful_extractions"] / len(request.urls) * 100 if request.urls else 0
                    },
                    "extractor_info": {
                        "type": "jina-ai+basic-cleaning",
                        "version": "2.1",
                        "endpoint": "/api/cdd/extract"
                    }
                },
                "extracted_content": []
            }
            
            # Convert extracted content to match your existing JSON format
            for i, result in enumerate(processed_results):
                jina_content = result.get("jina_content", {})
                raw_content = jina_content.get("content", "")
                cleaned_content = _basic_content_cleaning(raw_content)
                
                # Extract domain from URL
                parsed_url = urlparse(result.get("link", ""))
                source_domain = parsed_url.netloc
                
                content_entry = {
                    "index": i,
                    "link": result.get("link", ""),
                    "title": jina_content.get("title", result.get("title", "")),
                    "snippet": result.get("snippet", ""),
                    "source": source_domain,
                    "date": result.get("date", ""),
                    "source_query": result.get("source_query", "direct_url"),
                    "search_engine": result.get("search_engine", "direct"),
                    "extraction_status": result.get("extraction_status", "unknown"),
                    "jina_content": {
                        "title": jina_content.get("title", ""),
                        "content": cleaned_content,
                        "url": jina_content.get("url", result.get("link", "")),
                        "publishedTime": jina_content.get("publishedTime", ""),
                        "author": jina_content.get("author", ""),
                        "language": jina_content.get("language", "en"),
                        "description": jina_content.get("description", ""),
                        "keywords": jina_content.get("keywords", []),
                        "usage": jina_content.get("usage", {})
                    },
                    "extraction_metadata": {
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                        "content_length_original": len(jina_content.get("content", "")),
                        "content_length_cleaned": len(cleaned_content),
                        "source_domain": source_domain,
                        "extractor_version": "2.1"
                    }
                }
                
                json_data["extracted_content"].append(content_entry)
            
            # Write JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            # Calculate file size
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            logger.info(f"💾 Content extraction results saved to: {filepath}")
            logger.info(f"📁 File size: {file_size_mb:.2f} MB")
            logger.info(f"📊 Saved {len(json_data['extracted_content'])} content entries")
            
        except Exception as save_error:
            logger.error(f"⚠️ Failed to save JSON file: {save_error}")
            # Don't fail the entire request if saving fails
            pass
        
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
