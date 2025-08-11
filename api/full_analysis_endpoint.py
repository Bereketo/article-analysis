from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
import logging
import requests

router = APIRouter(
    prefix="/api/cdd",
    tags=["full-company-analysis"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post("/full-company-analysis")
def full_company_analysis(
    company_name: str = Query(...), 
    country: str = Query(...),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD")
):
    """
    Orchestrate full analysis for a company: aliases -> search -> extract -> article-analysis.
    Returns final output as JSON.
    """
    try:
        # 1. Generate aliases
        aliases_payload = {"company_name": company_name, "country": country}
        logger.info(f"Getting aliases for {company_name}")
        
        try:
            aliases_resp = requests.post("http://localhost:8000/api/cdd/aliases", json=aliases_payload)
            if aliases_resp.status_code != 200:
                raise HTTPException(status_code=aliases_resp.status_code, detail=f"Aliases error: {aliases_resp.text}")
            aliases_data = aliases_resp.json()
        except Exception as e:
            logger.error(f"Error in aliases step: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # 2. Perform search
        search_payload = {
            "primary_alias": aliases_data.get("primary_alias"),
            "aliases": aliases_data.get("aliases", []),
            "stock_symbols": aliases_data.get("stock_symbols", []),
            "local_variants": aliases_data.get("local_variants", []),
            "parent_company": aliases_data.get("parent_company"),
            "adverse_search_queries": aliases_data.get("adverse_search_queries", []),
            "all_aliases": aliases_data.get("all_aliases", ""),
            "confidence_score": aliases_data.get("confidence_score", 0),
            "total_adverse_queries": aliases_data.get("total_adverse_queries", 0),
            "start_date": start_date,
            "end_date": end_date
        }
        
        logger.info(f"Searching for articles")
        try:
            search_resp = requests.post("http://localhost:8000/api/cdd/search", json=search_payload)
            if search_resp.status_code != 200:
                raise HTTPException(status_code=search_resp.status_code, detail=f"Search error: {search_resp.text}")
            search_data = search_resp.json()
        except Exception as e:
            logger.error(f"Error in search step: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        # 3. Extract content from URLs
        simplified_data = search_data.get("simplified_data")
        if not simplified_data or not isinstance(simplified_data, dict):
            raise HTTPException(status_code=500, detail="Missing or malformed 'simplified_data' in search response")
        
        extract_payload = {
            "urls": simplified_data.get("urls", []), 
            "aliases": simplified_data.get("aliases", []),
            "parent_company_name": simplified_data.get("parent_company_name", "")
        }
        
        logger.info(f"Extracting content from {len(extract_payload['urls'])} URLs")
        try:
            extract_resp = requests.post("http://localhost:8000/api/cdd/extract", json=extract_payload)
            if extract_resp.status_code != 200:
                raise HTTPException(status_code=extract_resp.status_code, detail=f"Extract error: {extract_resp.text}")
            extract_data = extract_resp.json()
            extracted_articles = extract_data.get("extracted_content", [])
        except Exception as e:
            logger.error(f"Error in extract step: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        

        # 4. Analyze articles
        articles_for_analysis = [
            {
                "url": a.get("url"),
                "content": a.get("content"),
                "title": a.get("title")
            } for a in extracted_articles if a.get("url") and a.get("content")
        ]

        # Limit to 5 articles for analysis
        articles_for_analysis = articles_for_analysis

        simplified_data = extract_data.get("simplified_data", {})
        aliases = simplified_data.get("aliases") or extract_data.get("processing_summary", {}).get("aliases_used", [])
        parent_company_name = simplified_data.get("parent_company_name") or extract_data.get("processing_summary", {}).get("parent_company", "")

        analysis_payload = {
            "articles": articles_for_analysis,
            "aliases": aliases,
            "parent_company_name": parent_company_name
        }
        
        logger.info(f"Analyzing {len(articles_for_analysis)} articles")
        try:
            analysis_resp = requests.post("http://localhost:8000/api/cdd/article-analysis", json=analysis_payload)
            if analysis_resp.status_code != 200:
                raise HTTPException(status_code=analysis_resp.status_code, detail=f"Analysis error: {analysis_resp.text}")
            analysis_data = analysis_resp.json()
        except Exception as e:
            logger.error(f"Error in analysis step: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        logger.info("Full company analysis completed successfully")
        return analysis_data


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
