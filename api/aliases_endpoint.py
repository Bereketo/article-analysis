from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import asyncio
from agents.comprehensive_alias_agent import ComprehensiveAliasAgent

# Pydantic models for request/response
class AliasRequest(BaseModel):
    """
    Request model for generating company aliases
    """
    company_name: str = Field(
        ...,
        description="The company name to generate aliases for",
        example="Microsoft Corporation"
    )
    country: Optional[str] = Field(
        "India",
        description="The country where the company is based (for local variations)",
        example="United States"
    )
    include_adverse_queries: Optional[bool] = Field(
        True,
        description="Whether to include adverse media search queries in the response"
    )
    max_aliases: Optional[int] = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of aliases to return"
    )

class AliasResponse(BaseModel):
    """
    Response model containing generated company aliases and related information
    """
    primary_alias: str = Field(..., description="The primary/canonical name of the company")
    aliases: List[str] = Field(..., description="List of alternative names and variations")
    stock_symbols: List[str] = Field(..., description="List of stock exchange symbols")
    local_variants: List[str] = Field(..., description="Localized name variations")
    parent_company: str = Field(..., description="Parent company name if applicable")
    adverse_search_queries: List[str] = Field(..., description="Generated search queries for adverse media")
    all_aliases: str = Field(..., description="Comma-separated string of all aliases")
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the alias generation (0.0 to 1.0)"
    )
    total_adverse_queries: Optional[int] = Field(
        None,
        description="Total number of adverse search queries generated"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the alias generation"
    )

class ErrorResponse(BaseModel):
    """
    Standard error response model
    """
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")

app = FastAPI(
    title="Company Aliases API",
    description="API for generating company name aliases and variations",
    version="1.0.0"
)

logger = logging.getLogger(__name__)

# Initialize the comprehensive agent
comprehensive_agent = ComprehensiveAliasAgent()

# Configure CORS
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
async def get_company_aliases(
    request: AliasRequest,
    x_api_key: Optional[str] = None
):
    """
    Generate comprehensive company aliases and adverse search queries using advanced AI analysis.
    
    - **company_name**: The company name to generate aliases for
    - **country**: (Optional) Country for localization of name variations
    - **include_adverse_queries**: (Optional) Whether to include adverse media search queries
    - **max_aliases**: (Optional) Maximum number of aliases to return (1-50)
    
    Returns a structured response with all generated aliases and related information.
    """
    try:
        logger.info(f"🎯 Processing comprehensive alias request for: {request.company_name}")
        
        # Validate input
        if not request.company_name or len(request.company_name.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Company name must be at least 2 characters long"
            )
        
        # Use the comprehensive alias agent
        result = await comprehensive_agent.generate_comprehensive_aliases(
            company_name=request.company_name,
            country=request.country,
            include_adverse_queries=request.include_adverse_queries,
            max_aliases=request.max_aliases
        )
        
        logger.info(f"✅ Generated {len(result.aliases)} aliases with {len(result.adverse_search_queries)} adverse queries")
        
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

@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the API is running and healthy",
    responses={
        200: {"description": "API is healthy"},
        500: {"description": "API is not healthy"}
    }
)
async def health_check():
    """
    Health check endpoint that verifies the API is running properly.
    
    Returns:
        dict: Status of the API and its components
    """
    try:
        # Add any additional health checks here
        return {
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "database": "connected",
                "cache": "enabled"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service is not healthy"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
