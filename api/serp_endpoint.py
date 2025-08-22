from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import json
import os
import asyncio
import sys
from datetime import datetime, timedelta
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from services.json_to_excel_converter import convert_json_to_excel
from services.s3_service import upload_excel_to_s3
from services.email_service import SimpleEmailService

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
    download_url: Optional[str] = Field(None, description="Direct download link for Excel file", example="https://mybucket.s3.amazonaws.com/search-results/Company_results_20250814.xlsx")
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
    summary="Perform a web search with downloadable Excel results",
    description="""
    Execute a search query across multiple search engines and content types.
    
    The API will:
    1. Search across Google and DuckDuckGo with automatic deduplication
    2. Generate and save search results as JSON and Excel files locally
    3. Upload the Excel file to AWS S3 (public bucket) and provide direct download URLs
    4. Return search results with direct S3 download links
    
    **Download Options in Response:**
    - `download_url`: Direct public S3 URL that can be copied and used in any browser
    
    The download URL is a direct link to the Excel file stored in a public S3 bucket.
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
            concurrent_limit=24,
            use_duckduckgo=True  # Explicitly enable DuckDuckGo
        )
        
        # Get search results from both Google and DuckDuckGo with automatic deduplication
        search_results = extractor.extract_content(
            queries=request.adverse_search_queries,
            start_date=start_date,
            end_date=end_date,
            num_results=10
        )
        
        # Log search engine statistics before deduplication
        google_results_count = len([r for r in search_results if r.get('search_engine') == 'google'])
        duckduckgo_results_count = len([r for r in search_results if r.get('search_engine') == 'duckduckgo'])
        total_before_dedup = len(search_results)
        
        logger.info(f"📊 Search engine results before deduplication:")
        logger.info(f"   🔍 Google: {google_results_count} results")
        logger.info(f"   🦆 DuckDuckGo: {duckduckgo_results_count} results")
        logger.info(f"   📋 Total: {total_before_dedup} results")
        
        # Apply deduplication to remove duplicate URLs from Google and DuckDuckGo
        search_results = extractor._deduplicate_search_results(search_results)
        total_after_dedup = len(search_results)
        duplicates_removed = total_before_dedup - total_after_dedup
        
        logger.info(f"📊 After deduplication: {total_after_dedup} results ({duplicates_removed} duplicates removed)")

        # Prepare all search results for JSON serialization (no LLM filtering)
        serializable_results = []
        
        for result in search_results:
            serializable_result = {
                'title': result.get('title', ''),
                'link': result.get('link', ''),
                'snippet': result.get('snippet', ''),
                'source': result.get('source', '') if isinstance(result.get('source'), str) 
                        else result.get('source', {}).get('name', ''),
                'date': result.get('date', ''),
                'search_engine': result.get('search_engine', 'unknown'),
                'source_query': result.get('source_query', ''),
                'used_query': result.get('source_query', ''),  # NEW FIELD: Query used to find this article
                'search_period': result.get('search_period', {})
            }
            serializable_results.append(serializable_result)
        
        logger.info(f"✅ Processing completed with {len(serializable_results)} search results (no LLM filtering applied)")
        
        # Calculate total articles
        total_articles = len(serializable_results)
        
        # Create processing summary with search engine statistics
        processing_summary = {
            "queries_processed": len(request.adverse_search_queries),
            "aliases_used": len(request.aliases),
            "parent_company": request.parent_company,
            "total_articles_found": total_articles,
            "search_engine_statistics": {
                "google_results": google_results_count,
                "duckduckgo_results": duckduckgo_results_count,
                "total_before_deduplication": total_before_dedup,
                "total_after_deduplication": total_after_dedup,
                "duplicates_removed": duplicates_removed,
                "deduplication_rate": f"{(duplicates_removed/total_before_dedup*100):.1f}%" if total_before_dedup > 0 else "0%"
            }
        }

        # Extract URLs from search results
        extracted_urls = [result.get('link', '') for result in serializable_results if result.get('link')]
        
        # Format the response properly
        response = SerpResponse(
            download_url=None,  # Will be populated after S3 upload
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
        
        # Save search results to JSON file in search-results directory
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company_name_safe = request.parent_company.replace(" ", "_").replace("/", "_") if request.parent_company else "unknown_company"
            filename = f"search_results_{company_name_safe}_{timestamp}.json"
            
            # Ensure search-results directory exists
            search_results_dir = "search-results"
            os.makedirs(search_results_dir, exist_ok=True)
            
            # Full path for the JSON file
            file_path = os.path.join(search_results_dir, filename)
            
            # Prepare data to save (include all response data)
            save_data = {
                "timestamp": datetime.now().isoformat(),
                "request_summary": {
                    "primary_alias": request.primary_alias,
                    "parent_company": request.parent_company,
                    "total_queries": len(request.adverse_search_queries),
                    "queries": request.adverse_search_queries,
                    "aliases_count": len(request.aliases)
                },
                "results": {
                    "total_articles_found": total_articles,
                    "search_results": serializable_results,
                    "extracted_urls": extracted_urls,
                    "processing_summary": processing_summary
                }
            }
            
            # Write to JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Search results saved to: {file_path}")
            
            # Automatically generate Excel file from the JSON
            try:
                excel_filename = f"{company_name_safe}_search_results_{timestamp}.xlsx"
                excel_path = os.path.join(search_results_dir, excel_filename)
                
                logger.info(f"📊 Generating Excel file from JSON...")
                excel_result = convert_json_to_excel(
                    input_file=file_path, 
                    output_file=excel_path, 
                    no_style=False
                )
                
                if excel_result:
                    logger.info(f"📋 Excel file generated: {excel_result}")
                    response.processing_summary["excel_file"] = excel_result
                    
                    # Upload Excel file to S3 and get download URL
                    try:
                        logger.info(f"☁️ Uploading Excel file to S3...")
                        s3_result = upload_excel_to_s3(
                            file_path=excel_result,
                            custom_key=f"search-results/{excel_filename}"
                        )
                        
                        if s3_result["success"]:
                            logger.info(f"✅ Excel file uploaded to S3: {s3_result['s3_key']}")
                            response.download_url = s3_result["download_url"]
                            response.processing_summary["s3_upload"] = {
                                "success": True,
                                "s3_key": s3_result["s3_key"],
                                "bucket_name": s3_result["bucket_name"]
                            }
                        else:
                            logger.warning(f"⚠️ S3 upload failed: {s3_result['error']}")
                            response.processing_summary["s3_upload"] = {
                                "success": False,
                                "error": s3_result["error"]
                            }
                    except Exception as s3_error:
                        logger.error(f"❌ Error uploading to S3: {str(s3_error)}")
                        response.processing_summary["s3_upload"] = {
                            "success": False,
                            "error": f"S3 upload exception: {str(s3_error)}"
                        }
                    
                    # Send Excel results via email (using specialized search results method)
                    try:
                        logger.info(f"📧 Sending search results Excel via email...")
                        email_service = SimpleEmailService()
                        email_sent = email_service.send_search_results_excel(excel_result, request.parent_company)
                        if email_sent:
                            logger.info(f"✅ Search results Excel successfully sent via email")
                            response.processing_summary["email_sent"] = {
                                "success": True,
                                "recipient": email_service.recipient_email,
                                "sent_at": datetime.now().isoformat()
                            }
                        else:
                            logger.warning(f"⚠️ Failed to send search results Excel via email")
                            response.processing_summary["email_sent"] = {
                                "success": False,
                                "error": "Email sending failed"
                            }
                    except Exception as e:
                        logger.error(f"❌ Error sending search results email: {str(e)}")
                        response.processing_summary["email_sent"] = {
                            "success": False,
                            "error": str(e)
                        }
                        # Don't fail the whole request if email fails
                        pass
                else:
                    logger.warning(f"⚠️  Excel file generation failed")
                    
            except Exception as excel_error:
                logger.error(f"❌ Error generating Excel file: {str(excel_error)}")
                # Don't fail the request if Excel generation fails
            
            # Add file paths to response for reference
            response.processing_summary["saved_to_file"] = file_path
            
        except Exception as e:
            logger.error(f"❌ Error saving search results to file: {str(e)}")
            # Don't fail the request if file saving fails, just log it
        
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
