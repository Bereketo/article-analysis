from fastapi import APIRouter, HTTPException
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

router = APIRouter(
    prefix="/api/cdd",
    tags=["aliases"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# Initialize the comprehensive agent
comprehensive_agent = ComprehensiveAliasAgent()

@router.post(
    "/aliases", 
    response_model=AliasResponse,
    responses={
        200: {"description": "Successfully generated company aliases"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate company aliases",
    description="""
    Generate comprehensive company aliases, variations, and adverse media search queries.
    """
)
async def get_company_aliases(request: AliasRequest):
    try:
        logger.info(f"🎯 Processing comprehensive alias request for: {request.company_name}")
        
        result = await comprehensive_agent.generate_comprehensive_aliases(
            company_name=request.company_name,
            country=request.country
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error processing comprehensive alias request: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate comprehensive aliases: {str(e)}"
        )

@router.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "comprehensive-aliases-api",
        "agent_type": "comprehensive_alias_agent"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000)
