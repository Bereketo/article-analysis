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
            """Helper to extract from analysis['metadata'] if present, else from analysis - matching notebook logic"""
            analysis = result.get('analysis', {}) or {}
            metadata = analysis.get('metadata', {}) or {}
            if isinstance(metadata, dict) and field in metadata:
                return metadata.get(field, default)
            if isinstance(analysis, dict):
                return analysis.get(field, default)
            return default
        
        df_data = []
        for result in successful_results:
            analysis = result.get('analysis', {}) or {}
            content_meta = result.get('content_metadata', {})
            
            # Extract published date with robust parsing using the helper function
            published_date = (
                extract_metadata_field(result, 'published_date') or
                content_meta.get('published_date') or
                ''
            )
            
            # Convert to pandas timestamp if valid
            if published_date:
                try:
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
                'confidence_score': extract_metadata_field(result, 'confidence_score', 0.0),
                
                # Risk flags
                'has_fraud': extract_metadata_field(result, 'has_fraud', False),
                'has_litigation': extract_metadata_field(result, 'has_litigation', False), 
                'has_insolvency': extract_metadata_field(result, 'has_insolvency', False),
                'has_regulatory_action': extract_metadata_field(result, 'has_regulatory_action', False),
                
                # Author and metadata
                'author': extract_metadata_field(result, 'author', ''),
                'keywords': keywords,
                'source_query': result.get('source_query', ''),
                
                # Company relationship (matching notebook field names)
                'is_subsidiary_parent_company': extract_metadata_field(result, 'is_subsadariy_parent_company', False),
                'is_subsidiary_parent_company_reason': extract_metadata_field(result, 'is_subsadariy_parent_company_reason', ''),
                
                # Content length and timeline
                'content_length': content_meta.get('content_length', len(result.get('content', ''))),
                'event_timeline': extract_metadata_field(result, 'event_timeline', ''),
            }
            
            df_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Define export columns matching notebook structure exactly
        export_columns = [
            'url', 'title', 'published_date', 'is_adverse', 'is_adverse_reason',
            'risk_category', 'risk_explanation', 'risk_snippet', 'priority_level',
            'confidence_score', 'has_fraud', 'has_litigation', 'has_insolvency',
            'has_regulatory_action', 'author', 'keywords', 'source_query',
            'is_subsidiary_parent_company', 'is_subsidiary_parent_company_reason',
            'content_length', 'event_timeline'
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
        
        # Export to Excel with multiple sheets (exactly matching notebook)
        with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
            # Sheet 1: All cleaned articles
            df_cleaned[export_columns].to_excel(
                writer, 
                sheet_name='All_Articles', 
                index=False
            )
            
            # Sheet 2: Subsidiary-specific articles
            if len(subsidiary_articles) > 0:
                subsidiary_articles[export_columns].to_excel(
                    writer, 
                    sheet_name='Subsidiary_Specific', 
                    index=False
                )
            
            # Sheet 3: Parent company impact articles
            if len(parent_impact_articles) > 0:
                parent_impact_articles[export_columns].to_excel(
                    writer, 
                    sheet_name='Parent_Company_Impact', 
                    index=False
                )
            
            # Sheet 4: Adverse articles only
            if len(adverse_articles) > 0:
                adverse_articles[export_columns].to_excel(
                    writer, 
                    sheet_name='Adverse_Only', 
                    index=False
                )
            
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
        # Use the _create_analysis_prompt method
        prompt_messages = extractor._create_analysis_prompt(
            article.content,
            [article.url],  # Convert single URL to list
            aliases,
            parent_company_name
        )
        
        # Run LLM analysis
        response = await extractor.llm.ainvoke(prompt_messages)
        
        # Clean and parse the response
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
            "llm_model": "gpt-4.1",
            "analysis_version": "improved_content_extraction_agent",
            "processing_time_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        }
        
        # Prepare response
        result = {
            "url": article.url,
            "title": article.title,
            "content": article.content,
            "analysis": analysis,
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
        error_response = {
            "url": article.url,
            "title": article.title,
            "content": article.content,
            "analysis": {
                "is_filter": True,
                "is_filter_reason": f"Processing error: {str(e)}"
            },
            "risk_category": "error",
            "source": source,
            "content_metadata": {
                "content_length": len(article.content) if article.content else 0,
                "word_count": len(article.content.split()) if article.content else 0,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "llm_model": "gpt-4.1",
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
