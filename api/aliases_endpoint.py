from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import asyncio
import json
import re
from agents.alias_generation_improved import alias_agent_improved

# Pydantic models for request/response

logger = logging.getLogger(__name__)

class AliasRequest(BaseModel):
    company_name: str
    country: Optional[str] = "India"

async def _correct_spelling_and_validate(company_name: str, country: str) -> Dict[str, str]:
    """Correct spelling errors in company name and country using LLM"""
    
    correction_prompt = f"""
    You are a spelling correction assistant for corporate names and countries.
    
    Please correct any spelling errors in the following:
    Company Name: "{company_name}"
    Country: "{country}"
    
    Return the corrected versions in JSON format:
    {{
        "corrected_company_name": "Corrected Company Name",
        "corrected_country": "Corrected Country"
    }}
    
    Rules:
    - Fix obvious spelling mistakes
    - Standardize company suffixes (Ltd, Limited, Inc, etc.)
    - Use proper country names (not abbreviations)
    """
    try:
        from langchain_openai import AzureChatOpenAI
        from langchain.schema import HumanMessage
        import json
        import re
        import os
        
        llm = AzureChatOpenAI(
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]
        )
        
        response = await llm.ainvoke([HumanMessage(content=correction_prompt)])
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            corrections = json.loads(json_match.group())
            return {
                "company_name": corrections.get("corrected_company_name", company_name),
                "country": corrections.get("corrected_country", country)
            }
    except Exception as e:
        logger.warning(f"Spelling correction failed: {e}")
    
    # Fallback to original if correction fails
    return {"company_name": company_name, "country": country}

class AliasResponse(BaseModel):
    primary_alias: str = Field(description="Primary company name after research and correction")
    aliases: List[str] = Field(description="List of company aliases, variations, and former names")
    stock_symbols: List[str] = Field(description="Stock ticker symbols (NSE, BSE, NYSE, etc.)")
    local_variants: List[str] = Field(description="Regional or local name variations")
    parent_company: str = Field(description="Parent company or holding company name")
    target_names: List[str] = Field(description="Master list of all searchable company identifiers")
    adverse_search_queries: List[str] = Field(description="Full comprehensive set of adverse media search queries (~40-70 queries)")
    optimized_adverse_queries: List[str] = Field(description="Optimized set of high-impact adverse queries for fast 10-minute pipeline (~10 queries)")
    all_aliases: str = Field(description="Comma-separated string of all aliases")
    confidence_score: Optional[float] = Field(None, description="Confidence score of the alias generation (0.0-1.0)")
    total_adverse_queries: Optional[int] = Field(None, description="Total count of comprehensive adverse queries")
    total_optimized_queries: Optional[int] = Field(None, description="Total count of optimized adverse queries")

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
    
    Returns two sets of search queries:
    - `adverse_search_queries`: Full comprehensive set (~40-70 queries) for thorough analysis
    - `optimized_adverse_queries`: Optimized set (~10 queries) for fast 10-minute pipeline
    
    The optimized queries focus on the most critical adverse terms and use only the top 3 
    company identifiers to achieve 80-90% coverage in significantly less time.
    """
)
async def get_company_aliases(request: AliasRequest):
    try:
        logger.info(f"🎯 Processing comprehensive alias request")
        
        # Correct spelling errors
        corrections = await _correct_spelling_and_validate(request.company_name, request.country)
        corrected_company = corrections["company_name"]
        corrected_country = corrections["country"]
        if corrected_company != request.company_name or corrected_country != request.country:
            logger.info(f"📝 Spelling corrections applied: '{request.company_name}' → '{corrected_company}', '{request.country}' → '{corrected_country}'")

        alias_data = await alias_agent_improved.generate_aliases(corrected_company, corrected_country)
        logger.info(f"📊 Generated data - Primary: {alias_data['primary_alias']}, Aliases: {len(alias_data['aliases'])}, Adverse: {len(alias_data['adverse_search_queries'])}")
        
        # Structure the response
        return AliasResponse(
            primary_alias=alias_data["primary_alias"],
            aliases=alias_data["aliases"],
            stock_symbols=alias_data["stock_symbols"],
            local_variants=alias_data["local_variants"],
            parent_company=alias_data["parent_company"],
            target_names=alias_data["target_names"],
            adverse_search_queries=alias_data["adverse_search_queries"],
            optimized_adverse_queries=alias_data["optimized_adverse_queries"],  # NEW: Include optimized queries
            all_aliases=alias_data["all_aliases"],
            confidence_score=alias_data["confidence_score"],
            total_adverse_queries=len(alias_data["adverse_search_queries"]),
            total_optimized_queries=len(alias_data["optimized_adverse_queries"])  # NEW: Count of optimized queries
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing comprehensive alias request: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate comprehensive aliases: {str(e)}"
        )

@router.get("/aliases/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "comprehensive-aliases-api",
        "agent_type": "comprehensive_alias_agent"
    }

