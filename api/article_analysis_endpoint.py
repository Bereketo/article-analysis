from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import os
import asyncio
import time
from datetime import datetime, timezone, timedelta
import re
from tqdm.asyncio import tqdm
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent
from services.email_service import SimpleEmailService
from services.pdf_report_generator import AdverseMediaPDFReport

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
    # Include search metadata if available
    search_engine: Optional[str] = None
    search_query: Optional[str] = None
    published_date: Optional[str] = None
    source: Optional[str] = None

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
        
        # Process articles with progress tracking using tqdm
        
        logger.info(f"🔍 Starting analysis for {len(request.articles)} articles with progress tracking")
        start_time = time.time()
        
        # Process articles with progress tracking
        results = []
        progress_bar = tqdm(total=len(request.articles), desc="Analyzing Articles", unit="article")
        
        # Process articles in batches to show progress
        batch_size = 10  # Process 10 articles at a time
        successful_count = 0
        failed_count = 0
        
        for batch_start in range(0, len(request.articles), batch_size):
            batch_end = min(batch_start + batch_size, len(request.articles))
            batch_articles = request.articles[batch_start:batch_end]
            
            # Process batch in parallel
            batch_tasks = []
            for i, article in enumerate(batch_articles):
                actual_index = batch_start + i
                logger.debug(f"📄 Queueing article {actual_index+1}/{len(request.articles)}: {article.url or 'No URL provided'}")
                task = _process_single_article(
                    article=article,
                    extractor=extractor,
                    aliases=request.aliases,
                    parent_company_name=request.parent_company_name,
                    source=request.source
                )
                batch_tasks.append(task)
            
            # Wait for batch completion
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=300  # 5 minute timeout per batch
                )
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Batch {batch_start//batch_size + 1} timed out, continuing...")
                batch_results = [Exception("Timeout") for _ in batch_tasks]
            
            # Process batch results
            for i, result in enumerate(batch_results):
                actual_index = batch_start + i
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.error(f"❌ Error processing article {actual_index + 1}: {str(result)}")
                else:
                    successful_count += 1
                    results.append(result)
                
                # Update progress
                progress_bar.update(1)
                
                # Show progress every 50 articles
                if (actual_index + 1) % 50 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_article = elapsed_time / (actual_index + 1)
                    remaining_articles = len(request.articles) - (actual_index + 1)
                    estimated_remaining_time = avg_time_per_article * remaining_articles
                    
                    logger.info(f"📊 Progress Update:")
                    logger.info(f"   • Processed: {actual_index + 1}/{len(request.articles)} articles")
                    logger.info(f"   • Success: {successful_count}, Failed: {failed_count}")
                    logger.info(f"   • Elapsed time: {elapsed_time/60:.1f} minutes")
                    logger.info(f"   • Estimated remaining: {estimated_remaining_time/60:.1f} minutes")
                    logger.info(f"   • Current rate: {(actual_index + 1)/elapsed_time*60:.1f} articles/minute")
        
        progress_bar.close()
        
        # Final statistics
        total_time = time.time() - start_time
        logger.info(f"📊 FINAL ANALYSIS STATISTICS:")
        logger.info(f"   • Total articles processed: {len(request.articles)}")
        logger.info(f"   • Successful analyses: {successful_count}")
        logger.info(f"   • Failed analyses: {failed_count}")
        logger.info(f"   • Success rate: {(successful_count/len(request.articles)*100):.1f}%")
        logger.info(f"   • Total processing time: {total_time/60:.1f} minutes")
        logger.info(f"   • Average time per article: {total_time/len(request.articles):.2f} seconds")
        
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
        filename = f"{request.parent_company_name}_{timestamp_str}.json"
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

        # Convert results to DataFrame format
        def extract_metadata_field(result, field, default=None):
            """Helper to extract fields from raw LLM analysis structure for Excel export"""
            # Always use the processed analysis which we know works correctly
            analysis = result.get('analysis', {}) or {}
            
            # Handle case where analysis might be a list (shouldn't happen with new logic, but keep for safety)
            if isinstance(analysis, list):
                if analysis:
                    analysis = analysis[0]
                else:
                    return default
            
            if not isinstance(analysis, dict):
                logger.warning(f"Analysis is not a dict for field '{field}', got: {type(analysis)}")
                return default
            
            # For fields that should be at the top level of analysis (based on ArticleContent model)
            top_level_fields = {
                'author', 'keywords', 'is_filter', 'is_filter_reason', 
                'is_adverse', 'is_adverse_reason', 'published_date'
            }
            
            # For fields that should be in the metadata sub-object (based on ArticleMetadata model)
            metadata_fields = {
                'has_fraud', 'has_litigation', 'has_insolvency', 'has_regulatory_action',
                'risk_explanation', 'risk_snippet', 'priority_level', 'risk_category',
                'confidence_score', 'event_timeline', 'is_subsadariy_parent_company',
                'is_subsadariy_parent_company_reason'
            }
            
            # Check top level first for top-level fields
            if field in top_level_fields and field in analysis:
                logger.debug(f"Found '{field}' at top level: {analysis[field]}")
                return analysis[field]
            
            # Check metadata for metadata fields or as fallback
            metadata = analysis.get('metadata', {}) or {}
            # Handle case where metadata might be an empty array instead of object
            if isinstance(metadata, list):
                metadata = {}
            
            if isinstance(metadata, dict) and field in metadata:
                logger.debug(f"Found '{field}' in metadata: {metadata[field]}")
                return metadata[field]
            
            # Fallback to top level for any field not found in metadata
            if field in analysis:
                logger.debug(f"Found '{field}' at top level (fallback): {analysis[field]}")
                return analysis[field]
                
            logger.debug(f"Field '{field}' not found, using default: {default}")
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
            
            # Clean keywords field (convert list to string if needed)
            keywords = extract_metadata_field(result, 'keywords', [])
            if isinstance(keywords, list) and keywords:
                keywords = ", ".join([str(k) for k in keywords if k])
            elif not isinstance(keywords, str):
                keywords = ''
            
            author_or_source = ''
            # First try to get from the original article input
            original_article = None
            for orig_article in request.articles:
                if orig_article.url == result.get('url'):
                    original_article = orig_article
                    break
            
            if original_article and original_article.source:
                author_or_source = original_article.source
            else:
                # Fallback to LLM analysis
                author_or_source = extract_metadata_field(result, 'author', '')
                
                # Extract URL domain as final fallback if no author, but clean it up
                if not author_or_source:
                    try:
                        from urllib.parse import urlparse
                        parsed_url = urlparse(result.get('url', ''))
                        domain = parsed_url.netloc or ''
                        
                        # Clean up the domain to make it more readable
                        if domain:
                            # Remove www. prefix
                            if domain.startswith('www.'):
                                domain = domain[4:]
                            
                            # Convert common domains to readable names
                            domain_mapping = {
                                'timesofindia.indiatimes.com': 'Times of India',
                                'economictimes.indiatimes.com': 'The Economic Times',
                                'm.economictimes.com': 'The Economic Times',
                                'business-standard.com': 'Business Standard',
                                'financialexpress.com': 'The Financial Express',
                                'livemint.com': 'Mint',
                                'hindustantimes.com': 'Hindustan Times',
                                'indianexpress.com': 'The Indian Express',
                                'moneycontrol.com': 'Moneycontrol',
                                'businesstoday.in': 'Business Today',
                                'cnbctv18.com': 'CNBC TV18',
                                'ndtv.com': 'NDTV',
                                'thehindu.com': 'The Hindu',
                                'indiatoday.in': 'India Today',
                                'newindianexpress.com': 'The New Indian Express',
                                'deccanherald.com': 'Deccan Herald',
                                'tribuneindia.com': 'The Tribune',
                                'freepressjournal.in': 'Free Press Journal',
                                'outlookindia.com': 'Outlook India',
                                'news18.com': 'News18',
                                'abp.live': 'ABP Live',
                                'republicworld.com': 'Republic World',
                                'zeenews.india.com': 'Zee News',
                            }
                            
                            # Use mapped name if available, otherwise use cleaned domain
                            author_or_source = domain_mapping.get(domain, domain.replace('.com', '').replace('.in', '').title())
                        else:
                            author_or_source = ''
                            
                    except Exception:
                        author_or_source = ''
            
            # Extract date - use original search results date first
            article_date = ''
            if original_article and original_article.published_date:
                article_date = original_article.published_date
            else:
                # Fallback to LLM analysis
                article_date = extract_metadata_field(result, 'published_date', '')
            
            # Get risk assessment and conditionally extract risk reason
            risk_assessment = extract_metadata_field(result, 'is_adverse', 'Neutral')
            risk_reason = ''
            # Only populate risk reason for negative articles
            if risk_assessment == 'Negative':
                risk_reason = extract_metadata_field(result, 'is_adverse_reason', '')
            
            # Get search metadata from original input
            search_engine = ''
            search_query = ''
            if original_article:
                search_engine = original_article.search_engine or ''
                search_query = original_article.search_query or ''
            
            # Create row in Excel format (without Search Engine and Search Query Used)
            row = {
                'Company Name': request.parent_company_name,
                'Title': result.get('title', ''),
                'URL': result.get('url', ''),
                'Description': result.get('content', '')[:500] + '...' if len(result.get('content', '')) > 500 else result.get('content', ''),
                'Source': author_or_source,
                'Date': article_date,
                
                # Analysis-specific fields 
                'Risk Assessment': risk_assessment,
                'Risk Reason': risk_reason,  # Only populated for negative articles
                'Risk Category': extract_metadata_field(result, 'risk_category', ''),
                'Risk Explanation': extract_metadata_field(result, 'risk_explanation', ''),
                'Risk Snippet': extract_metadata_field(result, 'risk_snippet', ''),
                'Priority Level': extract_metadata_field(result, 'priority_level', ''),
                'Keywords': keywords,
                
                # Risk flags 
                'Has Fraud': extract_metadata_field(result, 'has_fraud', False),
                'Has Litigation': extract_metadata_field(result, 'has_litigation', False),
                'Has Insolvency': extract_metadata_field(result, 'has_insolvency', False),
                'Has Regulatory Action': extract_metadata_field(result, 'has_regulatory_action', False),
                
                # Parent/subsidiary relationship (preserve all)
                'Parent Company Impact': extract_metadata_field(result, 'is_subsadariy_parent_company', False),
                'Parent Company Impact Reason': extract_metadata_field(result, 'is_subsadariy_parent_company_reason', ''),
                
                # Filter status
                'Is Filtered': extract_metadata_field(result, 'is_filter', False),
                'Filter Reason': extract_metadata_field(result, 'is_filter_reason', ''),
            }
            
            df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Define export columns - remove Search Engine and Search Query Used since they're not meaningful
        export_columns = [
            # Core article data (6 columns instead of 8)
            'Company Name', 'Title', 'URL', 'Description', 'Source', 'Date',
            
            # Analysis-specific columns (preserve ALL fields)
            'Risk Assessment', 'Risk Reason', 'Risk Category', 'Risk Explanation', 'Risk Snippet', 'Priority Level',
            'Keywords', 'Has Fraud', 'Has Litigation', 'Has Insolvency', 'Has Regulatory Action',
            'Parent Company Impact', 'Parent Company Impact Reason', 'Is Filtered', 'Filter Reason'
        ]
        
        # Ensure all export columns exist
        for col in export_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Debug the dataframe before filtering
        logger.info(f"📊 DataFrame before filtering: {len(df)} rows")
        if len(df) > 0:
            logger.info(f"🔍 'Is Filtered' column values: {df['Is Filtered'].value_counts().to_dict()}")
            logger.info(f"🔍 Sample row data: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
        
        # Fix filtering logic - properly handle boolean values
        if 'Is Filtered' in df.columns and len(df) > 0:
            # Convert to proper boolean and filter out only True values
            df['Is Filtered'] = df['Is Filtered'].astype(bool)
            df_cleaned = df[~df['Is Filtered']].copy()  # Use ~ for NOT operator with boolean
            logger.info(f"📊 Articles after filtering: {len(df_cleaned)} rows (removed {len(df) - len(df_cleaned)} filtered articles)")
        else:
            df_cleaned = df.copy()
            logger.info(f"📊 No filtering applied, keeping all {len(df_cleaned)} articles")
        
        # Separate into segments using new column names
        subsidiary_articles = df_cleaned[df_cleaned['Parent Company Impact'] == False].copy()
        parent_impact_articles = df_cleaned[df_cleaned['Parent Company Impact'] == True].copy()
        adverse_articles = df_cleaned[df_cleaned['Risk Assessment'] == 'Negative'].copy()
        
        # Export to Excel with multiple sheets and styled headers
        try:
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
                    adverse_stats = df_cleaned['Risk Assessment'].value_counts()
                    for status, count in adverse_stats.items():
                        summary_data.append([f'{status} Articles', f'{count} ({count/len(df_cleaned)*100:.1f}%)'])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(
                    writer, 
                    sheet_name='Summary', 
                    index=False
                )
                _style_excel_headers(writer.book['Summary'])
                
        except Exception as excel_error:
            logger.error(f"❌ Error creating Excel file: {str(excel_error)}")
            # Try to create a basic Excel file without styling if the styled version fails
            try:
                with pd.ExcelWriter(excel_filepath, engine='openpyxl') as simple_writer:
                    df_cleaned[export_columns].to_excel(simple_writer, sheet_name='All_Articles', index=False)
                logger.info(f"✅ Created basic Excel file after styling failed")
            except Exception as simple_excel_error:
                logger.error(f"❌ Failed to create even basic Excel file: {str(simple_excel_error)}")
                raise HTTPException(status_code=500, detail=f"Excel generation failed: {str(simple_excel_error)}")
        
        # Export main dataset to CSV (matching notebook)
        try:
            df_cleaned[export_columns].to_csv(csv_filepath, index=False)
        except Exception as csv_error:
            logger.warning(f"⚠️ Failed to create CSV file: {str(csv_error)}")
            # Continue without CSV if it fails
        
        logger.info(f"📊 Excel analysis results written to {excel_filepath}")
        logger.info(f"📄 CSV analysis results written to {csv_filepath}")
        
        # Log summary statistics
        logger.info(f"Export Summary:")
        logger.info(f"    All Articles sheet: {len(df_cleaned)} rows")
        logger.info(f"   Subsidiary Specific sheet: {len(subsidiary_articles)} rows")
        logger.info(f"   Parent Company Impact sheet: {len(parent_impact_articles)} rows")
        logger.info(f"   Adverse Only sheet: {len(adverse_articles)} rows")
        logger.info(f"   Summary sheet: Analysis metadata")
        
        # Generate PDF Report for adverse findings
        pdf_report_path = None
        try:
            logger.info(f"📄 Generating PDF report for adverse media findings...")
            pdf_generator = AdverseMediaPDFReport()
            
            # Convert successful_results to format expected by PDF generator
            pdf_data = {
                'company_name': request.parent_company_name,
                'results': successful_results,
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Generate PDF report in same directory as Excel
            pdf_dir = os.path.join(output_dir, 'pdf-reports')
            pdf_report_path = await pdf_generator.generate_from_analysis_results(
                pdf_data,
                output_dir=pdf_dir
            )
            
            logger.info(f"✅ PDF report generated: {pdf_report_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate PDF report: {str(e)}")
            logger.debug(f"PDF generation error details:", exc_info=True)
            # Don't fail the whole request if PDF generation fails
            pdf_report_path = None
        
        # Send Excel results via email
        try:
            logger.info(f"📧 Sending Excel results via email...")
            email_service = SimpleEmailService()
            email_sent = email_service.send_excel_results(excel_filepath, request.parent_company_name)
            if email_sent:
                logger.info(f"✅ Excel results successfully sent via email")
            else:
                logger.warning(f"⚠️ Failed to send Excel results via email")
        except Exception as e:
            logger.error(f"❌ Error sending email: {str(e)}")
            # Don't fail the whole request if email fails
            pass
        
        # Add file paths to response for database tracking
        response_dict = response_data.model_dump()
        response_dict["file_paths"] = {
            "json": filepath,
            "excel": excel_filepath,
            "csv": csv_filepath,
            "pdf": pdf_report_path
        }
        
        return response_dict
        
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
        
        
        # Clean and parse the response using the existing logic for backward compatibility
        cleaned_content = clean_json_output(response.content)
        analysis = extractor.json_parser.parse(cleaned_content)
        
        
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
