"""
Full Analysis Pipeline with Database Tracking
=============================================

Enhanced version of full_analysis_endpoint.py with simple database integration.
REUSES all existing endpoint logic with minimal database additions.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import requests
import os
from datetime import datetime, timedelta

# Simple database service
from services.database_service import SimpleReportDB

class ContinueAnalysisRequest(BaseModel):
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
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # NEW: Optional database tracking
    report_uuid: Optional[str] = None
    report_name: Optional[str] = None
    notification_emails: Optional[List[str]] = []
    created_by: Optional[str] = "api"
    keyword_groups: Optional[Dict[str, List[str]]] = None

router = APIRouter(
    prefix="/api/cdd",
    tags=["full-analysis-with-db"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post("/create-report")
def create_report_and_start_analysis(
    company_name: str = Query(...), 
    country: str = Query(...),
    report_name: str = Query(...),
    notification_emails: List[str] = Query([]),
    created_by: str = Query("api"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD"),
    keyword_groups: Dict[str, List[str]] = Body(None, description="Keyword groups like {'Group 1': ['keyword1', 'keyword2'], 'Group 2': ['keyword3']}") 
):
    """
    Create report in database and start the analysis pipeline.
    Step 1: Creates database entry, then calls existing alias generation.
    """
    try:
        # Create database entry (Step 1)
        db = SimpleReportDB()
        report_uuid = db.create_report(
            report_name=report_name,
            company_name=company_name,
            alias_list=[company_name],  # Will be updated after alias generation
            notification_emails=notification_emails,
            created_by=created_by,
            start_date=start_date,
            end_date=end_date,
            keyword_groups=keyword_groups
        )
        
        # REUSE existing alias generation logic (no changes!)
        aliases_payload = {"company_name": company_name, "country": country}
        logger.info(f"Getting aliases for {company_name} (Report: {report_uuid})")
        
        try:
            aliases_resp = requests.post("http://localhost:8000/api/cdd/aliases", json=aliases_payload)
            if aliases_resp.status_code != 200:
                # Update database on failure
                db.update_status(report_uuid, "SEARCH_FAILED", "api", 
                               {"error": f"Aliases error: {aliases_resp.text}"})
                raise HTTPException(status_code=aliases_resp.status_code, detail=f"Aliases error: {aliases_resp.text}")
            aliases_data = aliases_resp.json()
        except Exception as e:
            db.update_status(report_uuid, "SEARCH_FAILED", "api", {"error": str(e)})
            logger.error(f"Error in aliases step: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Update database with generated aliases
        db.update_status(report_uuid, "SEARCH_IN_PROGRESS", "api", 
                        {"aliases_generated": True, "alias_count": len(aliases_data.get("aliases", []))})
        
        # Return aliases for user editing, but include report_uuid for tracking
        editable_response = {
            "report_uuid": report_uuid,  # NEW: Include report UUID for tracking
            "primary_alias": aliases_data.get("primary_alias"),
            "aliases": aliases_data.get("aliases", []),
            "stock_symbols": aliases_data.get("stock_symbols", []),
            "local_variants": aliases_data.get("local_variants", []),
            "parent_company": aliases_data.get("parent_company"),
            "target_names": aliases_data.get("target_names", []),
            "adverse_search_queries": aliases_data.get("adverse_search_queries", []),
            "all_aliases": aliases_data.get("all_aliases", ""),
            "confidence_score": aliases_data.get("confidence_score", 0.8),
            "total_adverse_queries": aliases_data.get("total_adverse_queries", None),
            "start_date": start_date,
            "end_date": end_date
        }
        
        logger.info(f"✅ Report {report_uuid} created and aliases generated for {company_name}")
        return editable_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/continue-with-tracking")
def continue_analysis_with_tracking(request: ContinueAnalysisRequest):
    """
    Continue analysis pipeline with database tracking.
    REUSES all existing endpoint logic from full_analysis_endpoint.py
    """
    db = SimpleReportDB()
    report_uuid = None
    
    try:
        # Extract report UUID if provided
        if request.report_uuid:
            report_uuid = request.report_uuid
        elif request.report_name and "uuid:" in request.report_name:
            report_uuid = request.report_name.replace("uuid:", "")
        
        logger.info(f"Continuing analysis with tracking (Report: {report_uuid})")
        
        # Step 2: Search (REUSE existing logic!)
        search_payload = {
            "primary_alias": request.primary_alias,
            "aliases": request.aliases,
            "stock_symbols": request.stock_symbols,
            "local_variants": request.local_variants,
            "parent_company": request.parent_company,
            "adverse_search_queries": request.adverse_search_queries,
            "all_aliases": request.all_aliases,
            "confidence_score": request.confidence_score,
            "total_adverse_queries": request.total_adverse_queries,
            "start_date": request.start_date,
            "end_date": request.end_date
        }
        
        logger.info(f"Searching for articles")
        try:
            search_resp = requests.post("http://localhost:8000/api/cdd/search", json=search_payload)
            if search_resp.status_code != 200:
                if report_uuid:
                    db.update_status(report_uuid, "SEARCH_FAILED", "api", 
                                   {"error": f"Search error: {search_resp.text}"})
                raise HTTPException(status_code=search_resp.status_code, detail=f"Search error: {search_resp.text}")
            search_data = search_resp.json()
            
            # Update database: search completed + extract and save missing data
            if report_uuid:
                search_summary = search_data.get("processing_summary", {})
                
                # Extract or generate start_date and end_date
                extracted_start_date = request.start_date
                extracted_end_date = request.end_date
                if not extracted_start_date or not extracted_end_date:
                    # Default: 1 year before to now
                    end_date_obj = datetime.now()
                    start_date_obj = end_date_obj - timedelta(days=365)
                    extracted_start_date = start_date_obj.strftime("%Y-%m-%d")
                    extracted_end_date = end_date_obj.strftime("%Y-%m-%d")
                
                # Extract keywords from the structured groups used in alias generation
                keyword_groups = {
                    "Group 1 - Fraud & Financial": ["fraud", "scandal", "kickback", "misconduct", "scam", "bribe", "corruption", 
                                                   "embezzlement", "money-laundering", "forgery", "default", "bankruptcy", 
                                                   "insolvency", "penalty", "fine"],
                    "Group 2 - Legal & Criminal": ["lawsuit", "litigation", "investigation", "probe", "arrested", "charged", 
                                                  "accused", "criminal", "police", "CBI", "murder", "rape", "assault", 
                                                  "violence", "terrorism"],
                    "Group 3 - Regulatory & Compliance": ["banned", "suspension", "sanctions", "violation", "breach", "imprisonment", 
                                                         "prison", "jail", "jailed", "sentenced", "conviction", "guilty", "manipulated", 
                                                         "manipulation", "compliance"]
                }
                
                # Get notification email from environment if not provided
                notification_emails = request.notification_emails or []
                if not notification_emails:
                    smtp_email = os.getenv('SMTP_RECIPIENT_EMAIL')
                    if smtp_email:
                        notification_emails = [smtp_email]
                
                # Update database with extracted data
                fields_to_update = {
                    'start_date': extracted_start_date,
                    'end_date': extracted_end_date,
                    'keyword_groups': keyword_groups,
                    'notification_emails': notification_emails
                }
                
                db.update_database_fields(report_uuid, fields_to_update)
                
                db.update_status(report_uuid, "SEARCH_COMPLETED", "search-worker", {
                    "found": search_summary.get("total_results", 0),
                    "deduped": search_summary.get("unique_results", 0)
                })
                
        except Exception as e:
            if report_uuid:
                db.update_status(report_uuid, "SEARCH_FAILED", "api", {"error": str(e)})
            logger.error(f"Error in search step: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Step 3: Content Extraction (REUSE existing logic!)
        simplified_data = search_data.get("simplified_data")
        if not simplified_data or not isinstance(simplified_data, dict):
            if report_uuid:
                db.update_status(report_uuid, "EXTRACTION_FAILED", "api", 
                               {"error": "Missing simplified_data in search response"})
            raise HTTPException(status_code=500, detail="Missing or malformed 'simplified_data' in search response")
        
        # Check if search returned any URLs
        urls = simplified_data.get("urls", [])
        if not urls or len(urls) == 0:
            error_msg = f"Search found 0 articles. No content to extract. Check search queries or date range."
            logger.warning(f"⚠️ {error_msg}")
            
            if report_uuid:
                db.update_status(report_uuid, "SEARCH_COMPLETED", "search-worker", {
                    "found": 0,
                    "warning": error_msg
                })
            
            # Return early with informative message instead of failing
            return {
                "status": "completed_with_warnings",
                "message": error_msg,
                "report_uuid": report_uuid,
                "results": [],
                "summary": {
                    "total_articles_processed": 0,
                    "successful_analyses": 0,
                    "failed_analyses": 0,
                    "risk_categories": {},
                },
                "total_articles": 0
            }
        
        # Update database: extraction started
        if report_uuid:
            db.update_status(report_uuid, "EXTRACTION_IN_PROGRESS", "extraction-worker")
        
        extract_payload = {
            "urls": urls, 
            "aliases": simplified_data.get("aliases", []),
            "parent_company_name": simplified_data.get("parent_company_name", "")
        }
        
        logger.info(f"Extracting content from {len(extract_payload['urls'])} URLs")
        try:
            extract_resp = requests.post("http://localhost:8000/api/cdd/extract", json=extract_payload)
            if extract_resp.status_code != 200:
                if report_uuid:
                    db.update_status(report_uuid, "EXTRACTION_FAILED", "extraction-worker", 
                                   {"error": f"Extract error: {extract_resp.text}"})
                raise HTTPException(status_code=extract_resp.status_code, detail=f"Extract error: {extract_resp.text}")
            extract_data = extract_resp.json()
            extracted_articles = extract_data.get("extracted_content", [])
            
            # Update database: extraction completed
            if report_uuid:
                extract_summary = extract_data.get("processing_summary", {})
                db.update_status(report_uuid, "EXTRACTION_COMPLETED", "extraction-worker", {
                    "extracted": extract_summary.get("total_urls_requested", 0),
                    "successful": extract_summary.get("successful_extractions", 0),
                    "failed": extract_summary.get("failed_extractions", 0)
                })
                
        except Exception as e:
            if report_uuid:
                db.update_status(report_uuid, "EXTRACTION_FAILED", "extraction-worker", {"error": str(e)})
            logger.error(f"Error in extract step: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        # Step 4: Analysis (REUSE existing logic!)
        # Update database: analysis started
        if report_uuid:
            db.update_status(report_uuid, "ANALYSIS_IN_PROGRESS", "analysis-worker")
        
        articles_for_analysis = [
            {
                "url": a.get("url"),
                "content": a.get("content"),
                "title": a.get("title")
            } for a in extracted_articles if a.get("url") and a.get("content")
        ]

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
                if report_uuid:
                    db.update_status(report_uuid, "ANALYSIS_FAILED", "analysis-worker", 
                                   {"error": f"Analysis error: {analysis_resp.text}"})
                raise HTTPException(status_code=analysis_resp.status_code, detail=f"Analysis error: {analysis_resp.text}")
            analysis_data = analysis_resp.json()
            
            # Update database: analysis completed
            if report_uuid:
                analysis_summary = analysis_data.get("summary", {})
                risk_categories = analysis_summary.get("risk_categories", {})
                
                db.update_status(report_uuid, "ANALYSIS_COMPLETED", "analysis-worker", {
                    "analyzed": analysis_summary.get("successful_analyses", 0),
                    "adverse": risk_categories.get("Negative", 0),
                    "risk_breakdown": risk_categories
                }, None, "report")  # File path would be set by analysis endpoint
                
        except Exception as e:
            if report_uuid:
                db.update_status(report_uuid, "ANALYSIS_FAILED", "analysis-worker", {"error": str(e)})
            logger.error(f"Error in analysis step: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        # Add report UUID and database fields to response for tracking
        if report_uuid:
            analysis_data["report_uuid"] = report_uuid
            analysis_data["database_tracking"] = True
            
            # Extract or generate dates
            start_date_final = request.start_date
            end_date_final = request.end_date
            if not start_date_final or not end_date_final:
                end_date_obj = datetime.now()
                start_date_obj = end_date_obj - timedelta(days=365)
                start_date_final = start_date_obj.strftime("%Y-%m-%d")
                end_date_final = end_date_obj.strftime("%Y-%m-%d")
            
            analysis_data["start_date"] = start_date_final
            analysis_data["end_date"] = end_date_final
            
            # Add notification emails
            notification_emails = request.notification_emails or []
            if not notification_emails:
                smtp_email = os.getenv('SMTP_RECIPIENT_EMAIL')
                if smtp_email:
                    notification_emails = [smtp_email]
            analysis_data["notification_emails"] = notification_emails
            
            # Add keyword groups from alias generation
            keyword_groups = {
                "Group 1 - Fraud & Financial": ["fraud", "scandal", "kickback", "misconduct", "scam", "bribe", "corruption", 
                                               "embezzlement", "money-laundering", "forgery", "default", "bankruptcy", 
                                               "insolvency", "penalty", "fine"],
                "Group 2 - Legal & Criminal": ["lawsuit", "litigation", "investigation", "probe", "arrested", "charged", 
                                              "accused", "criminal", "police", "CBI", "murder", "rape", "assault", 
                                              "violence", "terrorism"],
                "Group 3 - Regulatory & Compliance": ["banned", "suspension", "sanctions", "violation", "breach", "imprisonment", 
                                                     "prison", "jail", "jailed", "sentenced", "conviction", "guilty", "manipulated", 
                                                     "manipulation", "compliance"]
            }
            analysis_data["keyword_groups"] = keyword_groups

        logger.info("✅ Full company analysis with database tracking completed successfully")
        return analysis_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report-status/{report_uuid}")
def get_report_status(report_uuid: str):
    """Get current report status and progress"""
    try:
        db = SimpleReportDB()
        report_data = db.get_report(report_uuid)
        
        if not report_data:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "report_uuid": report_data["report_uuid"],
            "company_name": report_data["company_name"],
            "current_status": report_data["current_status"],
            "status_history": report_data["status_history"],
            "file_paths": {
                "article_list": report_data["article_list_file_path"],
                "article_details": report_data["article_details_file_path"],
                "report": report_data["report_file_path"]
            },
            "updated_at": report_data["updated_at"],
            "start_date": report_data["start_date"],
            "end_date": report_data["end_date"],
            "keyword_groups": report_data["keyword_groups"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports")
def list_all_reports():
    """List all reports with their current status"""
    try:
        import psycopg2
        from services.database_service import SimpleReportDB
        
        db = SimpleReportDB()
        with db._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT report_uuid, company_name, current_status, created_at, updated_at,
                           start_date, end_date, keyword_groups
                    FROM cdd_report_master 
                    ORDER BY updated_at DESC 
                    LIMIT 50
                """)
                
                reports = []
                for row in cur.fetchall():
                    reports.append({
                        "report_uuid": str(row[0]),
                        "company_name": row[1],
                        "current_status": row[2],
                        "created_at": row[3].isoformat() if row[3] else None,
                        "updated_at": row[4].isoformat() if row[4] else None,
                        "start_date": row[5],
                        "end_date": row[6],
                        "keyword_groups": row[7]
                    })
                
                return {"reports": reports, "count": len(reports)}
        
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# REUSE existing endpoints with minimal modifications
@router.post("/aliases-with-tracking")
def full_company_analysis_with_tracking(
    company_name: str = Query(...), 
    country: str = Query(...),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD")
):
    """
    Generate aliases with optional database tracking.
    Same as existing /aliases2 endpoint but with tracking capability.
    """
    try:
        # 1. REUSE existing alias generation (no changes!)
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
        
        # Return aliases_data for user editing (same as existing endpoint)
        editable_response = {
            "primary_alias": aliases_data.get("primary_alias"),
            "aliases": aliases_data.get("aliases", []),
            "stock_symbols": aliases_data.get("stock_symbols", []),
            "local_variants": aliases_data.get("local_variants", []),
            "parent_company": aliases_data.get("parent_company"),
            "target_names": aliases_data.get("target_names", []),
            "adverse_search_queries": aliases_data.get("adverse_search_queries", []),
            "all_aliases": aliases_data.get("all_aliases", ""),
            "confidence_score": aliases_data.get("confidence_score", 0.8),
            "total_adverse_queries": aliases_data.get("total_adverse_queries", None),
            "start_date": start_date,
            "end_date": end_date,
            # Add database tracking option
            "database_tracking_available": True
        }
        
        logger.info(f"Aliases generated for {company_name}, awaiting user input")
        return editable_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
