from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import os
from datetime import datetime
import re
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

def clean_json_output(json_str: str) -> str:
    """Clean common JSON formatting issues from LLM output"""
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Remove any text after the JSON block
    json_str = re.sub(r'\}.*$', '}', json_str, flags=re.DOTALL)
    return json_str

# Pydantic models for request/response
class ArticleAnalysisRequest(BaseModel):
    url: str
    content: str  # Add content as parameter
    aliases: List[str]
    parent_company_name: str
    source: Optional[str] = "direct"  # google, duckduckgo, or direct

class ArticleAnalysisResponse(BaseModel):
    url: str
    content: str
    analysis: Dict[str, Any]  # ArticleContent schema output
    risk_category: Optional[str]
    source: str
    alias: List[str]
    content_metadata: Dict[str, Any]
    timestamp: str

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

router = APIRouter(
    prefix="/api/cdd",
    tags=["article-analysis"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post(
    "/article-analysis",
    response_model=ArticleAnalysisResponse,
    responses={
        200: {"description": "Article analysis completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Analyze article content using LLM",
    description="""
    Analyze provided article content using the LLM analysis pipeline from _create_analysis_prompt.
    """
)
async def analyze_article(request: ArticleAnalysisRequest):
    try:
        logger.info(f"🔍 Starting article analysis for URL: {request.url}")
        
        # Initialize the content extraction agent
        extractor = ImprovedContentExtractionAgent(
            num_results=10,
            concurrent_limit=24
        )
        
        # Use the _create_analysis_prompt method directly
        prompt_messages = extractor._create_analysis_prompt(
            request.content, 
            request.url, 
            request.aliases, 
            request.parent_company_name
        )
        
        # Run LLM analysis
        try:
            response = await extractor.llm.ainvoke(prompt_messages)
            
            # Clean the JSON output before parsing
            cleaned_content = clean_json_output(response.content)
            analysis = extractor.json_parser.parse(cleaned_content)
            
            # Handle case where analysis might be a list
            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]
            elif isinstance(analysis, list) and len(analysis) == 0:
                analysis = {
                    "is_filter": True,
                    "is_filter_reason": "AI returned empty list - likely filtered content"
                }
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {request.url}: {str(e)}")
            analysis = {
                "is_filter": True,
                "is_filter_reason": f"Analysis failed: {str(e)}"
            }
        
        # Extract risk category from analysis
        risk_category = None
        if isinstance(analysis, dict) and analysis.get("metadata"):
            risk_category = analysis["metadata"].get("risk_category")
        elif isinstance(analysis, dict):
            risk_category = analysis.get("risk_category")
        
        content_metadata = {
            "content_length": len(request.content),
            "analyzed_at": datetime.now().isoformat(),
            "llm_model": "gpt-4.1",
            "analysis_version": "improved_content_extraction_agent"
        }
        
        # Prepare response
        response_data = ArticleAnalysisResponse(
            url=request.url,
            content=request.content,
            analysis=analysis,
            risk_category=risk_category,
            source=request.source,
            alias=request.aliases,
            content_metadata=content_metadata,
            timestamp=datetime.now().isoformat()
        )
        
        # Save to llm-analysis directory
        os.makedirs("llm-analysis", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"llm-analysis/article_analysis_{timestamp}.json"
        
        # Convert response to dict for JSON serialization
        output_data = {
            "url": response_data.url,
            "content": response_data.content,
            "analysis": response_data.analysis,
            "risk_category": response_data.risk_category,
            "source": response_data.source,
            "alias": response_data.alias,
            "content_metadata": response_data.content_metadata,
            "timestamp": response_data.timestamp
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Article analysis completed and saved to {filename}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error during article analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze article: {str(e)}"
        )

@router.get("/article-analysis/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "article-analysis-api",
        "agent_type": "improved_content_extraction_agent",
        "output_directory": "llm-analysis"
    }
