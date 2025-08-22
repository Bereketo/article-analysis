from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
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
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.0
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
    primary_alias: str
    aliases: List[str]
    stock_symbols: List[str]
    local_variants: List[str]
    parent_company: str
    target_names: List[str]
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
            all_aliases=alias_data["all_aliases"],
            confidence_score=alias_data["confidence_score"],
            total_adverse_queries=len(alias_data["adverse_search_queries"])
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

