from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import os
import asyncio
from datetime import datetime, timezone, timedelta
import re
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

def _style_excel_headers(worksheet):
    """Apply styling to Excel worksheet headers"""
    try:
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        # Define header styling
        header_font = Font(bold=True, color="FFFFFF", size=12)
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        # Define border
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Apply styling to header row (row 1)
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Set header row height
        worksheet.row_dimensions[1].height = 30
        
    except ImportError:
        # If openpyxl styling modules are not available, skip styling
        pass
    except Exception as e:
        # Log any styling errors but don't fail the export
        logger.warning(f"Failed to apply Excel styling: {str(e)}")

def clean_json_output(json_str: str) -> str:
    """Clean common JSON formatting issues from LLM output"""
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    # Remove any text after the JSON block
    json_str = re.sub(r'\}.*$', '}', json_str, flags=re.DOTALL)
    return json_str

# Pydantic models for request/response
class ArticleInput(BaseModel):
    url: str
    content: str
    title: Optional[str] = None

class ArticleAnalysisRequest(BaseModel):
    articles: List[ArticleInput]
    aliases: List[str]
    parent_company_name: str
    source: Optional[str] = "direct"  # google, duckduckgo, or direct

class ArticleAnalysisResult(BaseModel):
    url: str
    title: Optional[str] = None
    content: str
    analysis: Dict[str, Any]
    risk_category: Optional[str] = None
    source: str
    content_metadata: Dict[str, Any]
    timestamp: str

class ArticleAnalysisResponse(BaseModel):
    results: List[ArticleAnalysisResult]
    summary: Dict[str, Any]
    total_articles: int
    processed_at: str

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

router = APIRouter(
    prefix="/api/cdd",
    tags=["article-analysis"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post(
    "/article-analysis",
    response_model=ArticleAnalysisResponse,
    responses={
        200: {"description": "Article analysis completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Analyze multiple articles using LLM",
    description="""
    Analyze multiple articles using the LLM analysis pipeline.
    Takes a list of articles and processes them in parallel.
    """
)
async def analyze_articles(request: ArticleAnalysisRequest):
    try:
        logger.info(f"🔍 Starting analysis for {len(request.articles)} articles")
        
        # Validate input
        if not request.articles:
            logger.warning("No articles provided for analysis")
            return ArticleAnalysisResponse(
                results=[],
                summary={
                    "total_articles_processed": 0,
                    "successful_analyses": 0,
                    "failed_analyses": 0,
                    "risk_categories": {},
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat()
                },
                total_articles=0,
                processed_at=datetime.now(timezone.utc).isoformat()
            )
        
        # Initialize the content extraction agent
        try:
            extractor = ImprovedContentExtractionAgent(
                num_results=10,
                concurrent_limit=min(10, len(request.articles))  # Limit concurrent requests
            )
        except Exception as e:
            logger.error(f"Failed to initialize content extraction agent: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize content extraction agent: {str(e)}"
            )
        
        results = []
        risk_categories = {}
        
        # Process articles in parallel
        tasks = []
        for i, article in enumerate(request.articles):
            logger.info(f"📄 Queueing article {i+1}/{len(request.articles)}: {article.url or 'No URL provided'}")
            task = _process_single_article(
                article=article,
                extractor=extractor,
                aliases=request.aliases,
                parent_company_name=request.parent_company_name,
                source=request.source
            )
            tasks.append(task)
        
        # Gather all results with timeout
        try:
            logger.info(f"🚀 Processing {len(tasks)} articles in parallel...")
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300  # 5 minute timeout for all requests
            )
        except asyncio.TimeoutError:
            logger.error("Article analysis timed out after 5 minutes")
            raise HTTPException(
                status_code=504,
                detail="Article analysis timed out. Please try again with fewer articles or shorter content."
            )
        
        # Process results and collect statistics
        successful_results = []
        logger.info(f"Processing {len(results)} results from asyncio.gather")
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing article {i}: {str(result)}")
                logger.error(f"Exception type: {type(result).__name__}")
                if hasattr(result, '__traceback__'):
                    import traceback
                    logger.error("Traceback:" + ''.join(traceback.format_tb(result.__traceback__)))
                continue
            
            # Check if result is a dictionary (expected format)
            if not isinstance(result, dict):
                logger.error(f"Error processing article {i}: Expected dict but got {type(result).__name__}: {result}")
                continue
                
            logger.info(f"Successfully processed article {i}: {result.get('url', 'No URL')}")
            logger.debug(f"Article {i} result keys: {list(result.keys())}")
            
            # Ensure the result has all required fields
            required_fields = ["url", "content", "analysis", "source", "content_metadata", "timestamp"]
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                logger.warning(f"Article {i} is missing required fields: {missing_fields}")
                continue
                
            successful_results.append(result)
            
            # Update risk categories count
            risk = result.get('risk_category', 'unknown')
            risk_categories[risk] = risk_categories.get(risk, 0) + 1
            
        logger.info(f"Successfully processed {len(successful_results)} out of {len(results)} articles")
        
        # Prepare summary
        summary = {
            "total_articles_processed": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            "risk_categories": risk_categories,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Write the full response to a single JSON file in llm-analysis
        import hashlib
        output_dir = os.path.join(os.path.dirname(__file__), '../llm-analysis')
        os.makedirs(output_dir, exist_ok=True)
        # Use a timestamp and a hash of the parent_company_name + time for uniqueness
        hash_input = f"{request.parent_company_name}_{datetime.now(timezone.utc).isoformat()}"
        file_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:8]
        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        filename = f"article_analysis_{timestamp_str}_{file_hash}.json"
        filepath = os.path.join(output_dir, filename)
        response_data = ArticleAnalysisResponse(
            results=successful_results,
            summary=summary,
            total_articles=len(successful_results),
            processed_at=datetime.now(timezone.utc).isoformat()
        )
        # Write the response as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response_data.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"📝 Full analysis response written to {filepath}")

        # Write results to Excel file using notebook structure
        try:
            import pandas as pd
        except ImportError:
            import sys
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
            import pandas as pd

        excel_filename = filename.replace('.json', '.xlsx')
        excel_filepath = os.path.join(output_dir, excel_filename)
        csv_filepath = excel_filepath.replace('.xlsx', '.csv')

        # Convert results to DataFrame format matching notebook structure
        def extract_metadata_field(result, field, default=None):
            """Helper to extract fields from raw LLM analysis structure for Excel export"""
            # Use raw_analysis_for_excel instead of the processed analysis
            analysis = result.get('raw_analysis_for_excel', {}) or {}
            
            # Fallback to processed analysis if raw analysis is not available
            if not analysis or (isinstance(analysis, dict) and not analysis):
                analysis = result.get('analysis', {}) or {}
            
            # Handle case where analysis might be a list (shouldn't happen with new logic, but keep for safety)
            if isinstance(analysis, list):
                if analysis:
                    analysis = analysis[0]
                else:
                    return default
            
            # For fields that should be at the top level of analysis (based on ArticleContent model)
            top_level_fields = {
                'published_date', 'author', 'keywords', 'is_filter', 'is_filter_reason', 
                'is_adverse', 'is_adverse_reason'
            }
            
            # For fields that should be in the metadata sub-object (based on ArticleMetadata model)
            metadata_fields = {
                'has_fraud', 'has_litigation', 'has_insolvency', 'has_regulatory_action',
                'risk_explanation', 'risk_snippet', 'priority_level', 'risk_category',
                'confidence_score', 'event_timeline', 'is_subsadariy_parent_company',
                'is_subsadariy_parent_company_reason'
            }
            
            if not isinstance(analysis, dict):
                return default
                
            # Check top level first for top-level fields
            if field in top_level_fields and field in analysis:
                return analysis[field]
            
            # Check metadata for metadata fields or as fallback
            metadata = analysis.get('metadata', {}) or {}
            # Handle case where metadata might be an empty array instead of object
            if isinstance(metadata, list):
                metadata = {}
            
            if isinstance(metadata, dict) and field in metadata:
                return metadata[field]
            
            # Fallback to top level for any field not found in metadata
            if field in analysis:
                return analysis[field]
                
            return default
        
        df_data = []
        for result in successful_results:
            analysis = result.get('analysis', {}) or {}
            content_meta = result.get('content_metadata', {})
            
            # Debug: Log which analysis we're using for Excel export
            raw_analysis = result.get('raw_analysis_for_excel', {})
            if raw_analysis:
                logger.debug(f"Using raw analysis for Excel export for URL: {result.get('url', 'Unknown')}")
            else:
                logger.debug(f"Fallback to processed analysis for Excel export for URL: {result.get('url', 'Unknown')}")
            
            # Extract published date with robust parsing using the helper function
            published_date = (
                extract_metadata_field(result, 'published_date') or
                content_meta.get('published_date') or
                ''
            )
            
            # Skip Jina content extraction for performance
            
            # Convert to pandas timestamp if valid
            if published_date:
                try:
                    published_date = pd.to_datetime(published_date, errors='coerce')
                except:
                    published_date = ''
            
            # If still no date, try to extract from URL or other sources
            if not published_date:
                try:
                    # Try to get it from the article URL or title patterns
                    url = result.get('url', '')
                    title = result.get('title', '')
                    
                    # Look for date patterns in URL
                    date_pattern = r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})'
                    url_match = re.search(date_pattern, url)
                    if url_match:
                        year, month, day = url_match.groups()
                        published_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        published_date = pd.to_datetime(published_date, errors='coerce')
                except:
                    published_date = ''
            
            # Clean keywords field (convert list to string if needed)
            keywords = extract_metadata_field(result, 'keywords', [])
            if isinstance(keywords, list) and keywords:
                keywords = ", ".join([str(k) for k in keywords if k])
            elif not isinstance(keywords, str):
                keywords = ''
            
            row = {
                # Basic URL and content info (matching notebook)
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'published_date': published_date,
                'is_adverse': extract_metadata_field(result, 'is_adverse', 'Neutral'),
                'is_adverse_reason': extract_metadata_field(result, 'is_adverse_reason', ''),
                'risk_category': extract_metadata_field(result, 'risk_category', ''),
                'risk_explanation': extract_metadata_field(result, 'risk_explanation', ''),
                'risk_snippet': extract_metadata_field(result, 'risk_snippet', ''),
                'priority_level': extract_metadata_field(result, 'priority_level', ''),
                
                # Risk flags
                'has_fraud': extract_metadata_field(result, 'has_fraud', False),
                'has_litigation': extract_metadata_field(result, 'has_litigation', False), 
                'has_insolvency': extract_metadata_field(result, 'has_insolvency', False),
                'has_regulatory_action': extract_metadata_field(result, 'has_regulatory_action', False),
                
                # Author and metadata
                'author': extract_metadata_field(result, 'author', ''),
                'keywords': keywords,
                
                # Company relationship (matching notebook field names)
                'is_subsidiary_parent_company': extract_metadata_field(result, 'is_subsadariy_parent_company', False),
                'is_subsidiary_parent_company_reason': extract_metadata_field(result, 'is_subsadariy_parent_company_reason', ''),
                
                # Event timeline only
                'event_timeline': extract_metadata_field(result, 'event_timeline', ''),
            }
            
            df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Define export columns (removed content_length, confidence_score, and source_query)
        export_columns = [
            'url', 'title', 'published_date', 'is_adverse', 'is_adverse_reason',
            'risk_category', 'risk_explanation', 'risk_snippet', 'priority_level',
            'has_fraud', 'has_litigation', 'has_insolvency',
            'has_regulatory_action', 'author', 'keywords',
            'is_subsidiary_parent_company', 'is_subsidiary_parent_company_reason',
            'event_timeline'
        ]
        
        # Ensure all export columns exist
        for col in export_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Filter data (remove filtered articles if needed)
        df_cleaned = df.copy()
        
        # Separate into segments (matching notebook logic)
        subsidiary_articles = df_cleaned[df_cleaned['is_subsidiary_parent_company'] == False].copy()
        parent_impact_articles = df_cleaned[df_cleaned['is_subsidiary_parent_company'] == True].copy()
        adverse_articles = df_cleaned[df_cleaned['is_adverse'] == 'Negative'].copy()
        
        # Export to Excel with multiple sheets and styled headers
        with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
            # Sheet 1: All cleaned articles
            df_cleaned[export_columns].to_excel(
                writer, 
                sheet_name='All_Articles', 
                index=False
            )
            _style_excel_headers(writer.book['All_Articles'])
            
            # Sheet 2: Subsidiary-specific articles
            if len(subsidiary_articles) > 0:
                subsidiary_articles[export_columns].to_excel(
                    writer, 
                    sheet_name='Subsidiary_Specific', 
                    index=False
                )
                _style_excel_headers(writer.book['Subsidiary_Specific'])
            
            # Sheet 3: Parent company impact articles
            if len(parent_impact_articles) > 0:
                parent_impact_articles[export_columns].to_excel(
                    writer, 
                    sheet_name='Parent_Company_Impact', 
                    index=False
                )
                _style_excel_headers(writer.book['Parent_Company_Impact'])
            
            # Sheet 4: Adverse articles only
            if len(adverse_articles) > 0:
                adverse_articles[export_columns].to_excel(
                    writer, 
                    sheet_name='Adverse_Only', 
                    index=False
                )
                _style_excel_headers(writer.book['Adverse_Only'])
            
            # Sheet 5: Summary statistics (matching notebook)
            summary_data = [
                ['Company Name', request.parent_company_name],
                ['Parent Company', request.parent_company_name],
                ['Analysis Date', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')],
                ['Total Articles Processed', len(df)],
                ['Articles After Cleaning', len(df_cleaned)],
                ['Subsidiary-Specific Articles', len(subsidiary_articles)],
                ['Parent Company Impact Articles', len(parent_impact_articles)],
                ['Adverse Articles', len(adverse_articles)],
                ['', ''],
                ['Adverse Distribution', ''],
            ]
            
            # Add adverse statistics
            if len(df_cleaned) > 0:
                adverse_stats = df_cleaned['is_adverse'].value_counts()
                for status, count in adverse_stats.items():
                    summary_data.append([f'{status} Articles', f'{count} ({count/len(df_cleaned)*100:.1f}%)'])
            
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(
                writer, 
                sheet_name='Summary', 
                index=False
            )
            _style_excel_headers(writer.book['Summary'])
        
        # Export main dataset to CSV (matching notebook)
        df_cleaned[export_columns].to_csv(csv_filepath, index=False)
        
        logger.info(f"📊 Excel analysis results written to {excel_filepath}")
        logger.info(f"📄 CSV analysis results written to {csv_filepath}")
        
        # Log summary statistics
        logger.info(f"📈 Export Summary:")
        logger.info(f"   📄 All Articles sheet: {len(df_cleaned)} rows")
        logger.info(f"   🎯 Subsidiary Specific sheet: {len(subsidiary_articles)} rows")
        logger.info(f"   🏭 Parent Company Impact sheet: {len(parent_impact_articles)} rows")
        logger.info(f"   ⚠️  Adverse Only sheet: {len(adverse_articles)} rows")
        logger.info(f"   📊 Summary sheet: Analysis metadata")

        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error during article analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_single_article(
    article: ArticleInput,
    extractor: ImprovedContentExtractionAgent,
    aliases: List[str],
    parent_company_name: str,
    source: str = "direct"
) -> Dict[str, Any]:
    """Process a single article and return its analysis"""
    # Record start time for processing metrics
    start_time = datetime.now(timezone.utc)
    
    # Create a unique identifier for this article processing
    article_id = article.url or f"article_{start_time.timestamp()}"
    logger.info(f"🔍 Starting processing for article: {article_id}")
    logger.debug(f"Article content length: {len(article.content) if article.content else 0} chars")
    try:
        # Skip Jina extraction for performance - use only the provided content
        
        # Use the _create_analysis_prompt method
        prompt_messages = extractor._create_analysis_prompt(
            article.content,
            article.url,  # Pass URL as string, not list
            aliases,
            parent_company_name
        )
        
        # Run LLM analysis
        response = await extractor.llm.ainvoke(prompt_messages)
        
        # Store the raw LLM response for Excel export
        raw_llm_response = response.content
        
        # Parse the raw LLM response for Excel export (separate from processed analysis)
        raw_analysis_for_excel = None
        try:
            # Try to parse the raw response as JSON for Excel export
            if raw_llm_response and raw_llm_response.strip() and raw_llm_response != "[]":
                raw_analysis_for_excel = json.loads(raw_llm_response)
            else:
                # If raw response is empty or [], create a filtered placeholder
                raw_analysis_for_excel = {
                    "is_filter": True,
                    "is_filter_reason": "LLM returned empty or filtered response",
                    "metadata": {}
                }
        except json.JSONDecodeError as json_err:
            logger.warning(f"Failed to parse raw LLM response as JSON for Excel export: {json_err}")
            raw_analysis_for_excel = {
                "is_filter": True,
                "is_filter_reason": f"Invalid JSON response: {str(json_err)}",
                "metadata": {}
            }
        
        # DEBUG: Save raw LLM response to debug folder
        try:
            debug_dir = os.path.join(os.path.dirname(__file__), '../debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create a safe filename from URL
            safe_url = article.url or f"article_{start_time.timestamp()}"
            safe_url = re.sub(r'[^a-zA-Z0-9_-]', '_', safe_url)[:50]  # Limit length and sanitize
            timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:19]  # Remove microseconds
            debug_filename = f"raw_llm_response_{safe_url}_{timestamp_str}.json"
            debug_filepath = os.path.join(debug_dir, debug_filename)
            
            debug_data = {
                "article_url": article.url,
                "article_title": article.title,
                "content_length": len(article.content) if article.content else 0,
                "parent_company_name": parent_company_name,
                "aliases": aliases,
                "raw_llm_response": raw_llm_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_messages": [{
                    "role": msg.type if hasattr(msg, 'type') else "unknown",
                    "content": msg.content if hasattr(msg, 'content') else str(msg)
                } for msg in prompt_messages]
            }
            
            with open(debug_filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🐛 Raw LLM response saved to {debug_filepath}")
            
        except Exception as debug_error:
            logger.warning(f"Failed to save debug data: {str(debug_error)}")
        
        # Clean and parse the response using the existing logic for backward compatibility
        cleaned_content = clean_json_output(response.content)
        analysis = extractor.json_parser.parse(cleaned_content)
        
        # DEBUG: Save the parsed analysis results too
        try:
            debug_dir = os.path.join(os.path.dirname(__file__), '../debug')
            safe_url = article.url or f"article_{start_time.timestamp()}"
            safe_url = re.sub(r'[^a-zA-Z0-9_-]', '_', safe_url)[:50]
            timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:19]
            
            parsed_debug_filename = f"parsed_analysis_{safe_url}_{timestamp_str}.json"
            parsed_debug_filepath = os.path.join(debug_dir, parsed_debug_filename)
            
            parsed_debug_data = {
                "article_url": article.url,
                "cleaned_llm_content": cleaned_content,
                "parsed_analysis": analysis,
                "analysis_type": str(type(analysis).__name__),
                "analysis_is_list": isinstance(analysis, list),
                "analysis_is_dict": isinstance(analysis, dict),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            with open(parsed_debug_filepath, 'w', encoding='utf-8') as f:
                json.dump(parsed_debug_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"🐛 Parsed analysis saved to {parsed_debug_filepath}")
            
        except Exception as debug_error:
            logger.warning(f"Failed to save parsed analysis debug data: {str(debug_error)}")
        
        # Handle case where analysis might be a list
        if isinstance(analysis, list):
            if analysis:
                analysis = analysis[0]
            else:
                analysis = {
                    "is_filter": True,
                    "is_filter_reason": "No analysis results returned"
                }
        
        # Determine risk category
        risk_category = None
        if isinstance(analysis, dict):
            if analysis.get('is_filter', False):
                risk_category = "filtered"
            else:
                risk_category = analysis.get('risk_category')
        
        # Prepare content metadata
        content_metadata = {
            "content_length": len(article.content),
            "word_count": len(article.content.split()),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "llm_model": "gpt-4o",
            "analysis_version": "improved_content_extraction_agent",
            "processing_time_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        }
        
        # Prepare response
        result = {
            "url": article.url,
            "title": article.title,
            "content": article.content,
            "analysis": analysis,
            "raw_analysis_for_excel": raw_analysis_for_excel,  # Add the raw analysis for Excel export
            "raw_llm_response": raw_llm_response,  # Keep the raw response for reference
            "risk_category": risk_category,
            "source": source,
            "content_metadata": content_metadata,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        

        logger.info(f"✅ Successfully processed article: {article_id}")
        return result
        
    except Exception as e:
        error_msg = f"Error processing article {article_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Prepare error response
        error_analysis = {
            "is_filter": True,
            "is_filter_reason": f"Processing error: {str(e)}",
            "metadata": {}
        }
        
        error_response = {
            "url": article.url,
            "title": article.title,
            "content": article.content,
            "analysis": error_analysis,
            "raw_analysis_for_excel": error_analysis,  # Use same error analysis for Excel
            "raw_llm_response": "",  # Empty raw response for errors
            "risk_category": "error",
            "source": source,
            "content_metadata": {
                "content_length": len(article.content) if article.content else 0,
                "word_count": len(article.content.split()) if article.content else 0,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "llm_model": "gpt-4o",
                "analysis_version": "improved_content_extraction_agent",
                "error": str(e),
                "processing_time_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            },
            "timestamp": datetime.now(timezone.utc).isoformat()  
        }
        
        # For debugging - log the full error response
        logger.debug(f"Error response for {article_id}: {json.dumps(error_response, indent=2)}")
        
        # Re-raise the exception to be handled by the caller
        raise RuntimeError(f"Failed to process article {article_id}: {str(e)}") from e

@router.get("/article-analysis/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "article-analysis",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_type": "improved_content_extraction_agent",
        "output_directory": "llm-analysis"
    }
