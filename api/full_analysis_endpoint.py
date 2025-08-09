from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
import os
import json
from datetime import datetime, timezone
import hashlib
import httpx

router = APIRouter(
    prefix="/api/cdd",
    tags=["full-company-analysis"],
    responses={404: {"description": "Not found"}},
)

import requests

@router.post("/full-company-analysis")
def full_company_analysis(company_name: str = Query(...), country: str = Query(...)):
    """
    Orchestrate full analysis for a company: aliases -> search -> extract -> article-analysis.
    Returns final output as JSON.
    """
    try:
        # 1. Aliases (POST /api/cdd/aliases)
        aliases_payload = {"company_name": company_name, "country": country}
        print(f"[DEBUG] Sending to /api/cdd/aliases: {aliases_payload}")
        try:
            aliases_resp = requests.post("http://localhost:8000/api/cdd/aliases", json=aliases_payload)
            print(f"[DEBUG] /aliases status: {aliases_resp.status_code}")
            print(f"[DEBUG] /aliases raw response: {aliases_resp.text}")
        except Exception as e:
            print(f"[DEBUG] Exception during aliases API call: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        if aliases_resp.status_code != 200:
            raise HTTPException(status_code=aliases_resp.status_code, detail=f"Aliases error: {aliases_resp.text}")
        try:
            aliases_data = aliases_resp.json()
        except Exception as e:
            print(f"[DEBUG] Exception decoding aliases_resp.json(): {e}")
            raise HTTPException(status_code=500, detail=f"Exception decoding aliases_resp.json(): {e}")
        print(f"[DEBUG] ALIASES DATA: {aliases_data}")
        aliases = aliases_data.get("aliases", [])
        
        # 2. Search
        search_payload = {
            "primary_alias": "Adani Enterprises Limited",
            "aliases": [
                "Adani Enterprises Limited",
                "Adani Enterprises Ltd"
            ],
            "stock_symbols": [
                "ADANIENT"
            ],
            "local_variants": [
                "Adani Enterprises India",
                "Adani Enterprises Limited"
            ],
            "parent_company": "Adani Group",
            "adverse_search_queries": [
                "Adani Enterprises Limited fraud"
            ],
            "all_aliases": "Adani Enterprises Limited, Adani Enterprises Ltd",
            "confidence_score": 0.8,
            "total_adverse_queries": 0
            }

        """ 
        search_payload = {
            "primary_alias": aliases_data.get("primary_alias"),
            "aliases": aliases_data.get("aliases", []),
            "stock_symbols": aliases_data.get("stock_symbols", []),
            "local_variants": aliases_data.get("local_variants", []),
            "parent_company": aliases_data.get("parent_company"),
            "adverse_search_queries": aliases_data.get("adverse_search_queries", []),
            "all_aliases": aliases_data.get("all_aliases", ""),
            "confidence_score": aliases_data.get("confidence_score", 0),
            "total_adverse_queries": aliases_data.get("total_adverse_queries", 0)
        }
        """

        print(f"[DEBUG] Sending to /api/cdd/search: {search_payload}")
        try:
            search_resp = requests.post("http://localhost:8000/api/cdd/search", json=search_payload)
            print(f"[DEBUG] /search status: {search_resp.status_code}")
            print(f"[DEBUG] /search raw response: {search_resp.text}")
        except Exception as e:
            print(f"[DEBUG] Exception during search API call: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        if search_resp.status_code != 200:
            raise HTTPException(status_code=search_resp.status_code, detail=f"Search error: {search_resp.text}")
        try:
            search_data = search_resp.json()
        except Exception as e:
            print(f"[DEBUG] Exception decoding search_resp.json(): {e}")
            raise HTTPException(status_code=500, detail=f"Exception decoding search_resp.json(): {e}")
        print(f"[DEBUG] SEARCH DATA: {search_data}")
        """        
        """
        # 3. Extract
        # Dynamically extract urls, aliases, and parent_company_name from the 'simplified_data' key in search_data for /extract
        simplified_data = search_data.get("simplified_data")
        if not simplified_data or not isinstance(simplified_data, dict):
            print(f"[DEBUG] 'simplified_data' missing or malformed in /search response: {search_data}")
            raise HTTPException(status_code=500, detail="Missing or malformed 'simplified_data' in /search response")
        extract_payload = {
            "urls": simplified_data.get("urls", [])[:5],
            "aliases": simplified_data.get("aliases", []),
            "parent_company_name": simplified_data.get("parent_company_name", "")
        }
        print(f"[DEBUG] Sending to /api/cdd/extract: {extract_payload}")
        try:
            extract_resp = requests.post("http://localhost:8000/api/cdd/extract", json=extract_payload)
            print(f"[DEBUG] /extract status: {extract_resp.status_code}")
            print(f"[DEBUG] /extract raw response: {extract_resp.text}")
        except Exception as e:
            print(f"[DEBUG] Exception during extract API call: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        if extract_resp.status_code != 200:
            raise HTTPException(status_code=extract_resp.status_code, detail=f"Extract error: {extract_resp.text}")
        try:
            extract_data = extract_resp.json()
        except Exception as e:
            print(f"[DEBUG] Exception decoding extract_resp.json(): {e}")
            raise HTTPException(status_code=500, detail=f"Exception decoding extract_resp.json(): {e}")
        print(f"[DEBUG] EXTRACT DATA: {extract_data}")
        extracted_articles = extract_data.get("extracted_content", [])
        

        # 4. Article Analysis
        # Dynamically build articles for analysis with only url, content, title
        articles_for_analysis = [
            {
                "url": a.get("url"),
                "content": a.get("content"),
                "title": a.get("title")
            } for a in extracted_articles if a.get("url") and a.get("content")
        ]

        MAX_TEST_ARTICLES = 5  # or any small number you want
        articles_for_analysis = articles_for_analysis[:MAX_TEST_ARTICLES]

        simplified_data = extract_data.get("simplified_data", {})
        aliases = simplified_data.get("aliases") or extract_data.get("processing_summary", {}).get("aliases_used", [])
        parent_company_name = simplified_data.get("parent_company_name") or extract_data.get("processing_summary", {}).get("parent_company", "")

        analysis_payload = {
            "articles": articles_for_analysis,
            "aliases": aliases,
            "parent_company_name": parent_company_name
        }
        """
        temp_analysis_payload = {
            "articles": [
                {
                "url": "https://example.com/test-article-1",
                "content": "Reliance Industries Limited announced a major drawdown of its petrochemical operations. The company reported it lost over $2 billion so far. RIL's stock price went down following the announcement of the strategic partnership.",
                "title": "Reliance Industries Announces Major Petrochemical Restructuring",
                "source_domain": "example.com"
                },
                {
                "url": "https://example.com/test-article-2",
                "content": "In a surprising move, Reliance has decided to expand its retail operations while scaling back on petrochemicals. Analysts are divided on this strategy.",
                "title": "Reliance Shifts Focus from Petrochemicals to Retail",
                "source_domain": "example.com"
                }
            ],
            "aliases": [
                "Adani Enterprises Limited", "Adani Enterprises Ltd", "Adani Enterprises", "AEL", "Adani Exports Limited", "AE"
            ],
            "parent_company_name": "Adani group",
            "source": "direct"
            }
        """
        print(f"[DEBUG] Sending to /api/cdd/article-analysis: {analysis_payload}")
        try:
            analysis_resp = requests.post("http://localhost:8000/api/cdd/article-analysis", json=analysis_payload)
            print(f"[DEBUG] /article-analysis status: {analysis_resp.status_code}")
            print(f"[DEBUG] /article-analysis raw response: {analysis_resp.text}")
        except Exception as e:
            print(f"[DEBUG] Exception during article-analysis API call: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        if analysis_resp.status_code != 200:
            raise HTTPException(status_code=analysis_resp.status_code, detail=f"Analysis error: {analysis_resp.text}")
        try:
            analysis_data = analysis_resp.json()
        except Exception as e:
            print(f"[DEBUG] Exception decoding analysis_resp.json(): {e}")
            raise HTTPException(status_code=500, detail=f"Exception decoding analysis_resp.json(): {e}")
        print(f"[DEBUG] ANALYSIS DATA: {analysis_data}")

        # Return the analysis data directly, do not write to JSON or Excel here
        return analysis_data


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
