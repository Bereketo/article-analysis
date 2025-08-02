from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio
from agents.comprehensive_alias_agent import ComprehensiveAliasAgent

# Pydantic models for request/response
class AliasRequest(BaseModel):
    company_name: str
    country: Optional[str] = "India"

class AliasResponse(BaseModel):
    primary_alias: str
    aliases: List[str]
    stock_symbols: List[str]
    local_variants: List[str]
    parent_company: str
    adverse_search_queries: List[str]
    all_aliases: str
    confidence_score: Optional[float] = None
    total_adverse_queries: Optional[int] = None

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

app = FastAPI()
logger = logging.getLogger(__name__)

# Initialize the comprehensive agent
comprehensive_agent = ComprehensiveAliasAgent()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/aliases", 
    response_model=AliasResponse,
    responses={
        200: {"description": "Successfully generated company aliases"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["aliases"],
    summary="Generate company aliases",
    description="""
    Generate comprehensive company aliases, variations, and adverse media search queries.
    This endpoint uses advanced AI to analyze the company name and generate:
    - Common name variations
    - Localized name variants
    - Stock symbols
    - Parent company relationships
    - Adverse media search queries
    """
    )
async def get_company_aliases(request: AliasRequest):
    """
    Generate comprehensive company aliases and adverse search queries using advanced AI analysis
    
    Args:
        request: AliasRequest containing company_name and optional country
        
    Returns:
        AliasResponse with comprehensive alias information and extensive adverse search queries
    """
    
    try:
        logger.info(f"🎯 Processing comprehensive alias request for: {request.company_name}")
        
        # Use the comprehensive alias agent
        result = await comprehensive_agent.generate_comprehensive_aliases(
            company_name=request.company_name,
            country=request.country
        )
        
        logger.info(f"✅ Generated comprehensive aliases with {len(result.adverse_search_queries)} adverse queries")
        
        # Structure response
        response = AliasResponse(
            primary_alias=result.primary_alias,
            aliases=result.aliases,
            stock_symbols=result.stock_symbols,
            local_variants=result.local_variants,
            parent_company=result.parent_company,
            adverse_search_queries=result.adverse_search_queries,
            all_aliases=result.all_aliases,
            confidence_score=result.confidence_score,
            total_adverse_queries=len(result.adverse_search_queries)
        )
        
        logger.info(f"📊 Response: {len(response.aliases)} aliases, {len(response.adverse_search_queries)} adverse queries, confidence: {response.confidence_score}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing comprehensive alias request: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate comprehensive aliases: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "comprehensive-aliases-api",
        "agent_type": "comprehensive_alias_agent"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
