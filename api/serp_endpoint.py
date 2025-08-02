from fastapi import FastAPI, HTTPException, status, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional, Dict, Any, Union
import logging
import json
import os
import asyncio
from datetime import datetime
from enum import Enum
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

# Enums for search parameters
class SearchEngine(str, Enum):
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"

class ContentType(str, Enum):
    NEWS = "news"
    WEB = "web"
    IMAGES = "images"
    VIDEOS = "videos"

# Pydantic models for request/response
class SearchQuery(BaseModel):
    """Individual search query with optional parameters"""
    query: str = Field(..., description="The search query string")
    content_type: Optional[ContentType] = Field(
        ContentType.NEWS,
        description="Type of content to search for"
    )
    max_results: Optional[int] = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of results to return per query"
    )

class SerpRequest(BaseModel):
    """Request model for SERP API"""
    search_queries: List[SearchQuery] = Field(
        ...,
        description="List of search queries to execute"
    )
    aliases: List[str] = Field(
        ...,
        description="Company name aliases to include in the search"
    )
    parent_company_name: Optional[str] = Field(
        "Unknown",
        description="Name of the parent company for context"
    )
    search_engine: Optional[SearchEngine] = Field(
        SearchEngine.GOOGLE,
        description="Search engine to use for the query"
    )
    language: Optional[str] = Field(
        "en",
        description="Language code for search results (e.g., 'en', 'es', 'fr')"
    )
    region: Optional[str] = Field(
        "us",
        description="Region code for localized results (e.g., 'us', 'uk', 'ca')"
    )
    include_duplicates: Optional[bool] = Field(
        False,
        description="Whether to include duplicate results across different queries"
    )

class SearchResultItem(BaseModel):
    """Individual search result item"""
    title: str = Field(..., description="Title of the search result")
    url: HttpUrl = Field(..., description="URL of the search result")
    snippet: Optional[str] = Field(None, description="Short description/snippet")
    source: Optional[str] = Field(None, description="Source/domain of the result")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    content_type: Optional[ContentType] = Field(None, description="Type of content")
    relevance_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0 to 1.0)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the result"
    )

class ProcessingSummary(BaseModel):
    """Summary of search processing"""
    total_queries: int = Field(..., description="Total number of queries executed")
    total_results: int = Field(..., description="Total number of results found")
    unique_domains: int = Field(..., description="Number of unique domains in results")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    query_metrics: Dict[str, int] = Field(
        ...,
        description="Metrics per query (query -> result count)"
    )

class SerpResponse(BaseModel):
    """Response model for SERP API"""
    results: List[SearchResultItem] = Field(..., description="List of search results")
    total_results: int = Field(..., description="Total number of results")
    summary: ProcessingSummary = Field(..., description="Processing summary")
    search_metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata about the search execution"
    )

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")

# Initialize FastAPI app with metadata
app = FastAPI(
    title="SERP API",
    description="""
    Search Engine Results Page (SERP) API for performing web searches and extracting content.
    This API allows you to search across multiple search engines and content types.
    """,
    version="1.0.0"
)

# Configure logging
logger = logging.getLogger(__name__)

@app.post(
    "/search",
    response_model=SerpResponse,
    responses={
        200: {"description": "Search completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["search"],
    summary="Perform a web search",
    description="""
    Execute a search query across multiple search engines and content types.
    Returns structured search results with metadata and processing summary.
    """
)
async def search_content(
    request: SerpRequest,
    x_api_key: Optional[str] = None,
    x_request_id: Optional[str] = None
):
    """
    Perform a web search using the specified queries and parameters.
    
    - **search_queries**: List of search queries with parameters
    - **aliases**: Company name aliases to include in search
    - **parent_company_name**: (Optional) Parent company name
    - **search_engine**: (Optional) Search engine to use (default: google)
    - **language**: (Optional) Language code (default: en)
    - **region**: (Optional) Region code (default: us)
    - **include_duplicates**: (Optional) Include duplicate results (default: false)
    
    Returns structured search results with metadata and processing information.
    """
    try:
        logger.info(f"🔍 Starting content search with {len(request.search_queries)} queries")
        
        # Validate input
        if not request.search_queries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one search query is required"
            )
        
        # Initialize the content extraction agent with request parameters
        extractor = ImprovedContentExtractionAgent(
            num_results=request.search_queries[0].max_results if request.search_queries else 10,
            concurrent_limit=24
        )
        
        # First, get just the search results
        search_results = extractor.extract_content(
            queries=request.search_queries,
            num_results=10
        )
        
        # Save search results to a JSON file
        os.makedirs("search_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_results_file = f"search_results/search_results_{timestamp}.json"
        
        # Initialize Azure OpenAI LLM for title analysis
        from langchain_openai import AzureChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = AzureChatOpenAI(
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4.1",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )
        
        # System message to guide the LLM's analysis
        company_name = request.parent_company_name or "the company"
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
        
        # Save to JSON file
        with open(search_results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'search_queries': request.search_queries,
                'aliases': request.aliases,
                'parent_company_name': request.parent_company_name,
                'timestamp': datetime.now().isoformat(),
                'result_count': len(serializable_results),
                'results': serializable_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(serializable_results)} search results to {search_results_file}")
        
        # Calculate total articles
        total_articles = len(serializable_results)
        
        # Create processing summary
        processing_summary = {
            "queries_processed": len(request.search_queries),
            "aliases_used": len(request.aliases),
            "parent_company": request.parent_company_name,
            "total_articles_found": total_articles
        }
        
        response = SerpResponse(
            results_data={
                "search_results": serializable_results
            },
            total_articles=total_articles,
            processing_summary=processing_summary
        )
        
        logger.info(f"✅ Content search completed: {total_articles} articles found")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error during content search: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to perform content search: {str(e)}"
        )

@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the SERP API service is running and healthy",
    responses={
        200: {"description": "API is healthy"},
        500: {"description": "API is not healthy"}
    }
)
async def health_check():
    """
    Health check endpoint that verifies the SERP API service is running properly.
    
    Returns:
        dict: Status of the API and its components
    """
    try:
        # Add any additional health checks here
        return {
            "status": "healthy",
            "service": "serp-api",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "search_engine": "connected",
                "content_extractor": "ready"
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
        "api.serp_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )