from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import os
import asyncio
from datetime import datetime, timedelta
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

# Pydantic models for request/response
class SerpRequest(BaseModel):
    primary_alias: str
    aliases: List[str]
    stock_symbols: List[str]
    local_variants: List[str]
    parent_company: str
    adverse_search_queries: List[str]
    all_aliases: str
    confidence_score: Optional[float] = None
    total_adverse_queries: Optional[int] = None
    start_date: Optional[str] = None  # Format: "YYYY-MM-DD"
    end_date: Optional[str] = None    # Format: "YYYY-MM-DD"
    time_window_days: Optional[int] = None  # Alternative: last N days

class SimplifiedSerpData(BaseModel):
    urls: List[str]
    aliases: List[str]
    parent_company_name: str

class SerpResponse(BaseModel):
    results_data: Dict[str, Any]
    total_articles: int
    processing_summary: Dict[str, Any]
    simplified_data: SimplifiedSerpData

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

router = APIRouter(
    prefix="/api/cdd",
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
        logger.info(f"🔍 Starting content search with {len(request.adverse_search_queries)} queries")
        
        # Validate search queries
        if not request.adverse_search_queries:
            raise HTTPException(
                status_code=422,
                detail="Search queries list cannot be empty"
            )
        
        # Process time window
        start_date_str, end_date_str = _process_time_window(request.start_date, request.end_date, request.time_window_days)
        start_date = None
        end_date = None
        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            logger.info(f"📅 Search time window: {start_date_str} to {end_date_str}")

        extractor = ImprovedContentExtractionAgent(
            num_results=10,
            concurrent_limit=24
        )
        
        
        # First, get just the search results
        search_results = extractor.extract_content(
            queries=request.adverse_search_queries,
            start_date=start_date,
            end_date=end_date,
            num_results=10
        )


        
        # Initialize Azure OpenAI LLM for title analysis
        from langchain_openai import AzureChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = AzureChatOpenAI(
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )
        
        # System message to guide the LLM's analysis
        company_name = request.parent_company or "the company"
        system_prompt = f"""
        You are an AI assistant that analyzes news article titles to determine if they are specifically about {company_name} and relevant to adverse events or negative news.
        
        STRICT RULES:
        1. The article must be specifically about {company_name} or its direct subsidiaries
        2. The article must discuss actual adverse events, not just mention the company name
        3. Exclude articles that only mention the company in passing or in a list
        4. Exclude articles about other companies with similar names
        5. Exclude articles that are about the company's competitors or unrelated businesses
        
        Focus on these types of risks:
        - Financial (fraud, accounting issues, losses)
        - Legal (lawsuits, regulatory actions, fines)
        - Reputational (scandals, controversies)
        - Operational (safety incidents, major failures)
        - Regulatory (compliance violations, sanctions)
        
        Respond with a JSON object containing:
        {{
            "is_relevant": boolean,  // true only if the title is specifically about {company_name} and contains adverse content
            "reason": string,        // brief explanation of your decision
            "risk_category": string  // 'financial', 'legal', 'reputational', 'regulatory', 'operational', or 'none' if not relevant
        }}
        """
        
        # Prepare search results for JSON serialization
        serializable_results = []
        
        async def analyze_title(title):
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Analyze this news title: {title}")
                ]
                response = await llm.ainvoke(messages)
                return json.loads(response.content)
            except Exception as e:
                logger.error(f"Error analyzing title '{title}': {str(e)}")
                return {"is_relevant": False, "reason": "Error in analysis", "risk_category": "none"}
        
        # Process titles in batches for better performance
        batch_size = 10
        for i in range(0, len(search_results), batch_size):
            batch = search_results[i:i + batch_size]
            
            # Analyze all titles in the current batch
            analysis_results = await asyncio.gather(
                *[analyze_title(result.get('title', '')) for result in batch]
            )
            
            # Process results
            for result, analysis in zip(batch, analysis_results):
                if analysis.get('is_relevant', False):
                    serializable_result = {
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'source': result.get('source', '') if isinstance(result.get('source'), str) 
                                else result.get('source', {}).get('name', ''),
                        'date': result.get('date', ''),
                        'search_engine': result.get('search_engine', 'unknown'),
                        'source_query': result.get('source_query', ''),
                        'search_period': result.get('search_period', {}),
                        'relevance_analysis': {
                            'is_relevant': analysis.get('is_relevant', False),
                            'reason': analysis.get('reason', ''),
                            'risk_category': analysis.get('risk_category', 'none')
                        }
                    }
                    serializable_results.append(serializable_result)
                    logger.info(f"Included article: {result.get('title')} - {analysis.get('risk_category')}")
                else:
                    logger.debug(f"Excluding article - Not relevant: {result.get('title')}")
                    
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(search_results):
                await asyncio.sleep(1)
        
        logger.info(f"✅ Processing completed with {len(serializable_results)} search results")
        
        # Calculate total articles
        total_articles = len(serializable_results)
        
        # Create processing summary
        processing_summary = {
            "queries_processed": len(request.adverse_search_queries),
            "aliases_used": len(request.aliases),
            "parent_company": request.parent_company,
            "total_articles_found": total_articles
        }


        # Extract URLs from search results
        extracted_urls = [result.get('link', '') for result in serializable_results if result.get('link')]
        
        # Format the response properly
        response = SerpResponse(
            simplified_data=SimplifiedSerpData(
                urls=extracted_urls,
                aliases=request.aliases,
                parent_company_name=request.parent_company
            ),
            results_data={
                "search_results": serializable_results
            },
            total_articles=total_articles,
            processing_summary=processing_summary

        )
        
        logger.info(f"✅ Search completed. Found {total_articles} articles")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error during content search: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to perform content search: {str(e)}"
        )

def _process_time_window(start_date: str, end_date: str, time_window_days: int) -> tuple:
    """Process time window parameters and return start/end dates"""
    
    try:
        if start_date and end_date:
            # Use provided date range
            return start_date, end_date
        elif time_window_days:
            # Calculate date range from days
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=time_window_days)).strftime("%Y-%m-%d")
            return start_date, end_date
        else:
            # No time filter
            return None, None
    except Exception as e:
        logger.warning(f"Time window processing failed: {e}")
        return None, None


@router.get("/serp/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "serp-content-search-api",
        "agent_type": "improved_content_extraction_agent"
    }
