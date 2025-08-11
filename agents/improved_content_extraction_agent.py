import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from dateutil.relativedelta import relativedelta
from tqdm.asyncio import tqdm
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import logging
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
import os
import sys

os.environ["AZURE_OPENAI_API_KEY"] = "1f5d1bb6920844248ea17f61f73f82ac"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-gpt-echo.openai.azure.com"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"

# Configure logging for Jupyter notebooks
def setup_logging():
    """Setup logging configuration that works in Jupyter notebooks"""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Create a stream handler that outputs to stdout (visible in notebooks)
    handler = logging.StreamHandler(sys.stdout)
    
    # Create a detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Set up the logger
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # Also set up our specific logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

# Initialize logging
logger = setup_logging()

# Set up Azure OpenAI environment variables
os.environ["AZURE_OPENAI_API_KEY"] = "1f5d1bb6920844248ea17f61f73f82ac"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-gpt-echo.openai.azure.com"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"

class ArticleMetadata(BaseModel):
    # Adverse Event Detection
    has_fraud: bool = Field(
        default=False, description="Indicates if fraud is detected in the article"
    )
    has_litigation: bool = Field(
        default=False, description="Indicates if litigation is mentioned in the article"
    )
    has_insolvency: bool = Field(
        default=False, description="Indicates if insolvency issues are detected"
    )
    has_regulatory_action: bool = Field(
        default=False, description="Indicates if regulatory actions are mentioned"
    )
    risk_explanation: Optional[str] = Field(
        None, description="Short explanation justifying the risk classification"
    )
    risk_snippet: Optional[str] = Field(
        None, description="Text snippet from article supporting risk classification"
    )

    # Risk Metadata
    priority_level: Optional[Literal["High", "Medium", "Low"]] = Field(
        default=None,
        description="Priority level of the risk (High, Medium, Low) or None if not applicable",
    )
    risk_category: Literal[
        "Legal",
        "Financial",
        "Compliance",
        "Operational",
        "Reputational",
        "Strategic",
        "Other",
    ] = Field(
        default="Other",
        description="Category of risk (Legal, Financial, Compliance, etc.)",
    )
    confidence_score: Optional[float] = Field(
        default=0.0, description="Confidence score of the risk detection (0.0-1.0)"
    )
    event_timeline: Optional[str] = Field(
        None, description="Timeline of the risk event"
    )
    is_subsadariy_parent_company: bool = Field(
        default=False,
        description="True if the article is talking about a parent company issue that impacts the subsidiary/target company, or if both are discussed and there is a correlation or impact. False if the article is talking only and explicitly about the subsidiary/target company, with no impact or connection to the parent company.",
    )
    is_subsadariy_parent_company_reason: Optional[str] = Field(
        None, description="Reason for the is_subsadariy_parent_company classification"
    )

class ArticleContent(BaseModel):
    metadata: ArticleMetadata
    published_date: Optional[str] = Field(
        None, description="Publication date of the article in the format YYYY-MM-DD"
    )
    author: Optional[str] = Field(None, description="Author of the article")
    keywords: Optional[List[str]] = Field(
        None, description="Keywords extracted from the article"
    )
    is_filter: bool = Field(
        default=False,
        description="Indicates if this article should be filtered as it is not related to the company",
    )
    is_filter_reason: Optional[str] = Field(
        None, description="Reason for the filter classification"
    )
    is_adverse: Literal["Negative", "Neutral", "Positive"] = Field(
        default="Neutral",
        description="Indicates if this article is having any bad news related to the company, based on sentiment analysis",
    )
    is_adverse_reason: Optional[str] = Field(
        None, description="Reason for the adverse classification"
    )

class ImprovedContentExtractionAgent:
    def __init__(self, num_results=50, concurrent_limit=24, use_duckduckgo=True):
        self.num_results = num_results
        self.concurrent_limit = concurrent_limit
        self.use_duckduckgo = use_duckduckgo
        self.jina_api_key = "jina_80ed8ae9b65c4f25b71edc336f7cbfc07A5rTYKwAFLa1Hpy-33VAsQ8cLPY"
        
        # Initialize LangChain LLM
        self.llm = AzureChatOpenAI(
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o",  # Updated to use gpt-4o deployment
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )
        
        # Initialize JSON parser
        self.json_parser = JsonOutputParser()
        
        logger.info("🚀 Initializing ImprovedContentExtractionAgent with fresh extraction and analysis")
        logger.info(f"📊 Configuration: concurrent_limit={concurrent_limit}, num_results={num_results}, use_duckduckgo={use_duckduckgo}")
        
        # Google search configuration
        self.google_params = {
            "engine": "google",
            "google_domain": "google.co.in",
            "gl": "in",
            "hl": "en",
            "cr": "countryIN",
            "lr": "lang_en",
            "tbm": "nws",
            "num": num_results,
        }

        self.google_search = SerpAPIWrapper(
            serpapi_api_key="f055086d4349703c4f399e24eb7db6a54b37eb130c2330a2dea1ead04381ccb5",
            params=self.google_params,
        )
        
        # DuckDuckGo search configuration (if enabled)
        if self.use_duckduckgo:
            self.duckduckgo_params = {
                "engine": "duckduckgo",
                "kl": "in-en",  # India English
                "safe": "-1",   # Moderate filtering
            }
            
            self.duckduckgo_search = SerpAPIWrapper(
                serpapi_api_key="f055086d4349703c4f399e24eb7db6a54b37eb130c2330a2dea1ead04381ccb5",
                params=self.duckduckgo_params,
            )
            
            logger.info("🔍 DuckDuckGo search enabled")
        else:
            self.duckduckgo_search = None
            logger.info("🔍 DuckDuckGo search disabled")
        
        # For backward compatibility
        self.search = self.google_search

    def _normalize_url(self, url):
        """Normalize URL for better deduplication"""
        if not url:
            return ""

        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc

            # Remove www prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Remove common country/region subdomains
            country_subdomains = {
                "in.", "us.", "uk.", "ca.", "au.", "de.", "fr.", "es.", "it.", "br.", "mx.",
                "jp.", "kr.", "cn.", "hk.", "sg.", "my.", "th.", "ph.", "id.", "vn.", "tw.",
                "ru.", "pl.", "nl.", "be.", "ch.", "at.", "se.", "no.", "dk.", "fi.", "ie.",
                "pt.", "gr.", "cz.", "hu.", "ro.", "bg.", "hr.", "si.", "sk.", "lt.", "lv.",
                "ee.", "za.", "eg.", "ae.", "sa.", "il.", "tr.", "ar.", "cl.", "co.", "pe.",
                "ve.", "uy.", "py.", "bo.", "ec.", "gt.", "cr.", "pa.", "do.", "cu.", "pr.",
                "jm.", "ht.", "hn.", "sv.", "ni.", "bz.",
            }

            for subdomain in country_subdomains:
                if domain.startswith(subdomain):
                    domain = domain[len(subdomain):]
                    break

            # Remove common tracking parameters
            tracking_params = {
                "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                "fbclid", "gclid", "msclkid", "ref", "source", "campaign_id", "_ga", "_gid",
                "mc_cid", "mc_eid", "hl", "gl", "lang",
            }

            query_params = parse_qs(parsed.query)
            filtered_params = {
                k: v for k, v in query_params.items() 
                if k.lower() not in tracking_params
            }

            sorted_query = urlencode(sorted(filtered_params.items()), doseq=True)
            path = parsed.path.rstrip("/")
            if not path:
                path = "/"

            normalized = urlunparse((
                parsed.scheme, domain, path, parsed.params, sorted_query, ""
            ))

            return normalized

        except Exception:
            return url

    def _deduplicate_search_results(self, search_results):
        """Remove duplicate search results using normalized URL comparison"""
        seen_normalized = set()
        unique_results = []
        duplicates_found = 0

        for result in search_results: 
            url = result.get("link", "")
            normalized = self._normalize_url(url)
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_results.append(result)
            else:
                duplicates_found += 1

        logger.info(
            f"Search results deduplication: {len(search_results)} -> {len(unique_results)} ({duplicates_found} duplicates removed)"
        )
        return unique_results

    def _get_date_range_params(self, start_date, end_date, relative_df=None, engine="google"):
        """Generate date range parameters for search engines"""
        if engine == "duckduckgo":
            params = {**self.duckduckgo_params}
            if relative_df:
                params["df"] = relative_df
            else:
                # DuckDuckGo uses YYYY-MM-DD..YYYY-MM-DD format
                params["df"] = (
                    f"{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
                )
        else:  # google
            params = {**self.google_params}
            params["tbs"] = (
                f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}"
            )
        return params

    async def extract_content_with_jina(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Extract content from URL using Jina AI endpoint with retry mechanism for 503 errors"""
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}",
            "X-Retain-Images": "none"
        }
        
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(jina_url, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get("data", {})
                        elif response.status == 503:
                            if attempt < max_retries:
                                wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 3, 5 seconds
                                logger.warning(f"Jina API returned 503 for {url}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries + 1})")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.error(f"Failed to extract content from {url}: HTTP 503 after {max_retries + 1} attempts")
                                return {}
                        else:
                            logger.error(f"Failed to extract content from {url}: HTTP {response.status}")
                            return {}
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"Error extracting content from {url}: {str(e)}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error extracting content from {url} after {max_retries + 1} attempts: {str(e)}")
                    return {}
        
        return {}

    def _create_analysis_prompt(self, content: str, url: str, aliases: List[str], parent_company_name: str) -> List:
        """Create analysis prompt for a single article"""
        system_prompt = f"""You are assisting in risk analysis for media articles. The goal is to assess the relevance and tone of an article related to the company aliases {aliases} and  its parent company {parent_company_name}. Follow the guidelines below to evaluate content and return structured metadata when applicable.

        ⸻

        🔍 Relevance Assessment

        Determine if the article primarily concerns {aliases} or is materially related to {parent_company_name}:

            • If the article solely discusses {parent_company_name} without referencing or impacting {aliases}, mark:
                • is_filter = true
                • is_filter_reason = "Focus is only on the parent company; no relevance to the target company."

            • If both {parent_company_name} and {aliases} are mentioned but without meaningful connection, mark:
                • is_filter = true
                • is_filter_reason = "Mentions are independent; no clear relationship or effect."

            • If a clear connection or impact is evident between the entities, or if the article focuses on {aliases}, consider it relevant.

            • Common exclusions:
                - Brief mention of {aliases} with no consequence
                - Articles centered on unrelated topics or companies
                - Background-only or historical mentions without current context

            • If content is general (e.g., background info or company overview) with no newsworthy event or development, mark:
                • is_filter = true
                • is_filter_reason = "General context; no specific development or incident."

            • Consider an article relevant if:
                - It centers on {aliases}' activities (legal, strategic, operational, reputational)
                - It involves {parent_company_name} or a related entity in a way that has clear implications for {aliases}

        ⸻

        🧠 Content Analysis (For Relevant Articles Only)

        ⸻

        ✅ Step 1: Detect Adverse Events

        Identify negative developments only if they demonstrably affect the company (e.g., financial loss, penalties, backlash, failures, key exits).

        ⸻

        ✅ Step 2: Assess Tone

        Assign one of the following to `is_adverse`:
            • "Negative" – if harm or critical tone is present
            • "Positive" – if the article reflects growth, progress, or favorable developments
            • "Neutral" – if the article is informational and balanced

        Provide a concise explanation under `is_adverse_reason`.

        ⸻

        ✅ Step 3: Risk Metadata (If is_adverse = "Negative")

        If risk is identified, extract:
            • risk_explanation
            • risk_snippet
            • risk_category (choose from: "Legal", "Financial", "Compliance", "Operational", "Reputational", "Strategic", "Other")
            • priority_level ("High", "Medium", "Low")
            • confidence_score (float between 0.0–1.0)
            • event_timeline (if date is mentioned)
            • is_subsadariy_parent_company:
                - True if parent-company issues affect the subsidiary
                - False if only the subsidiary is discussed
            • is_subsadariy_parent_company_reason

        Limit to the most significant risk if multiple exist.

        ⸻

        ✅ Step 4: General Metadata

        Extract:
            • published_date (YYYY-MM-DD): Look carefully for dates in the article content, headers, or URL. Search for patterns like "January 15, 2023", "15/01/2023", "2023-01-15", or similar date formats. If no explicit publication date is found, try to infer from context clues or timestamps. Return null only if absolutely no date information is available.
            • author or source
            • keywords (3–7 terms summarizing the core topic)

        ⸻

        🧾 Output Format
        
        ALWAYS respond with a valid JSON object following the {ArticleContent.model_json_schema()} schema.
        
        For relevant articles: Fill out all applicable fields based on the content analysis.
        For irrelevant articles: Set is_filter=true and provide is_filter_reason, leave other fields as defaults or null.
        
        Never return an empty array or any other format - always return a complete JSON object.

        """

        human_prompt = f"""
        Analyze the article content below:

        URL: {url}
        Content: {content}
        please only return the json format described above.
        """

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]


    async def extract_and_analyze_single_url(self, search_result: Dict[str, Any], aliases: List[str], parent_company_name: str) -> Dict[str, Any]:
        """Extract content and analyze a single URL - always fresh processing"""
        url = search_result.get("link", "")
        
        try:
            # Step 1: Extract content from Jina AI
            logger.debug(f"Extracting content for {url}")
            jina_content = await self.extract_content_with_jina(url)
            
            # Determine extraction status
            if not jina_content.get("content", ""):
                search_result["extraction_status"] = "no_content"
            else:
                search_result["extraction_status"] = "success"
            
            search_result["jina_content"] = jina_content
            
            # Check if we have content to analyze
            content = jina_content.get("content", "")
            if not content:
                search_result["analysis"] = {
                    "is_filter": True,
                    "is_filter_reason": "No content extracted from URL - skipping AI analysis"
                }
                return search_result
            
            # Step 2: Run LLM analysis
            logger.debug(f"Running LLM analysis for {url}")
            prompt_messages = self._create_analysis_prompt(content, url, aliases, parent_company_name)
            
            try:
                response = await self.llm.ainvoke(prompt_messages)
                analysis = self.json_parser.parse(response.content)
                
                # Handle case where analysis might be a list
                if isinstance(analysis, list) and len(analysis) > 0:
                    analysis = analysis[0]
                elif isinstance(analysis, list) and len(analysis) == 0:
                    analysis = {
                        "is_filter": True,
                        "is_filter_reason": "AI returned empty list - likely filtered content"
                    }
                
                search_result["analysis"] = analysis
                
            except Exception as e:
                logger.error(f"Error in AI analysis for {url}: {str(e)}")
                analysis = {
                    "is_filter": True,
                    "is_filter_reason": f"Analysis failed: {str(e)}"
                }
                search_result["analysis"] = analysis
            
            return search_result
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            search_result["jina_content"] = {}
            search_result["extraction_status"] = "failed"
            search_result["extraction_error"] = str(e)
            search_result["analysis"] = {
                "is_filter": True,
                "is_filter_reason": f"Content extraction failed: {str(e)}"
            }
            
            return search_result

    async def extract_and_analyze_parallel(self, search_results: List[Dict[str, Any]], aliases: List[str], parent_company_name: str) -> List[Dict[str, Any]]:
        """
        Extract content and analyze in parallel - single step approach
        Both Jina AI extraction and OpenAI analysis happen together for each URL
        """
        logger.info(f"🚀 Starting fresh extraction and analysis for {len(search_results)} URLs")
        logger.info(f"📊 Configuration: concurrent_limit={self.concurrent_limit}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        async def process_single_url(search_result: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single URL with semaphore control"""
            async with semaphore:
                return await self.extract_and_analyze_single_url(search_result, aliases, parent_company_name)
        
        # Process all URLs in parallel
        tasks = [process_single_url(result) for result in search_results]
        
        # Use tqdm for progress tracking
        results = []
        for coro in tqdm.as_completed(tasks, desc="Processing URLs"):
            result = await coro
            results.append(result)
        
        # Generate summary statistics
        total_processed = len(results)
        successful_extractions = sum(1 for r in results if r.get("extraction_status") == "success")
        no_content_extractions = sum(1 for r in results if r.get("extraction_status") == "no_content")
        failed_extractions = sum(1 for r in results if r.get("extraction_status") == "failed")
        
        def get_analysis_dict(result):
            """Helper function to safely extract analysis as dictionary"""
            analysis = result.get("analysis", {})
            if isinstance(analysis, list) and len(analysis) > 0:
                return analysis[0]
            elif isinstance(analysis, list):
                return {}
            return analysis
        
        # Count AI analysis results
        actually_analyzed = sum(1 for r in results 
                              if r.get("extraction_status") == "success" 
                              and "analysis" in r 
                              and not any(skip_reason in (get_analysis_dict(r).get("is_filter_reason") or "") 
                                        for skip_reason in ["no content", "extraction failed", "Analysis failed"]))
        
        ai_filtered_articles = sum(1 for r in results 
                                 if get_analysis_dict(r).get("is_filter", False) 
                                 and r.get("extraction_status") == "success"
                                 and "no content" not in (get_analysis_dict(r).get("is_filter_reason") or "")
                                 and "extraction failed" not in (get_analysis_dict(r).get("is_filter_reason") or ""))
        
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"   📋 Total URLs processed: {total_processed}")
        logger.info(f"   🔄 Content Extraction Results:")
        logger.info(f"      ✅ Successful extractions: {successful_extractions}")
        logger.info(f"      📭 No content found: {no_content_extractions}")
        logger.info(f"      ❌ Failed extractions: {failed_extractions}")
        logger.info(f"   🤖 AI Analysis Results:")
        logger.info(f"      🔍 Successfully analyzed by AI: {actually_analyzed}")
        logger.info(f"      🏷️  AI-filtered as irrelevant: {ai_filtered_articles}")
        logger.info(f"   📈 Overall success rate: {(successful_extractions/total_processed)*100:.1f}%")
        
        return results

    def extract_content(self, queries, start_date=None, end_date=None, relative_df=None, num_results=50):
        """Extract relevant content based on queries using both Google and DuckDuckGo"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()

        all_search_results = []
        current_date = start_date

        logger.info(f"Starting content extraction for queries: {queries}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Num results per query: {num_results}")
        logger.info(f"Search engines: Google + {'DuckDuckGo' if self.use_duckduckgo else 'None'}")

        # Calculate number of windows (months)
        total_months = (
            (end_date.year - start_date.year) * 12
            + (end_date.month - start_date.month)
            + 1
        )

        window_pbar = tqdm(total=total_months, desc="Date Windows", unit="window")
        
        while current_date < end_date:
            # Calculate window end date (1 month from current date)
            window_end = min(current_date + relativedelta(months=1), end_date)

            logger.info(f"Searching for period: {current_date.date()} to {window_end.date()}")

            for query in tqdm(queries, desc=f"Queries ({current_date.date()} to {window_end.date()})", leave=False):
                logger.info(f"Searching query: {query}")
                
                # Search with Google
                try:
                    google_date_params = self._get_date_range_params(
                        current_date, window_end, relative_df=relative_df, engine="google"
                    )
                    self.google_search.params = google_date_params
                    
                    google_response = self.google_search.results(query)
                    google_results = google_response.get("news_results", [])
                    logger.info(f"Google: Found {len(google_results)} results for query '{query}'")

                    for result in google_results:
                        result["source_query"] = query
                        result["search_engine"] = "google"
                        result["search_period"] = {
                            "start": current_date.isoformat(),
                            "end": window_end.isoformat(),
                        }
                    all_search_results.extend(google_results)

                except Exception as e:
                    logger.error(f"Error searching Google for query '{query}': {e}")
                
                # Search with DuckDuckGo (if enabled) - with pagination
                if self.use_duckduckgo and self.duckduckgo_search:
                    try:
                        ddg_date_params = self._get_date_range_params(
                            current_date, window_end, relative_df=relative_df, engine="duckduckgo"
                        )
                        
                        # Calculate number of pages needed for DuckDuckGo
                        # Each page returns 10 results, but we want num_results total
                        pages_needed = max(1, (num_results + 9) // 10)  # Ceiling division
                        ddg_results = []
                        
                        logger.info(f"DuckDuckGo: Fetching {pages_needed} pages for query '{query}' (target: {num_results} results)")
                        
                        for page in range(pages_needed):
                            # Calculate start offset for this page (0, 1, 2, etc.)
                            start_offset = page
                            
                            # Update params for this page
                            page_params = ddg_date_params.copy()
                            page_params["start"] = start_offset
                            
                            self.duckduckgo_search.params = page_params
                            
                            try:
                                ddg_response = self.duckduckgo_search.results(query)
                                page_results = ddg_response.get("organic_results", [])
                                
                                logger.info(f"DuckDuckGo page {page + 1} (start={start_offset}): Found {len(page_results)} results")
                                
                                # Add metadata to each result and normalize field names
                                for result in page_results:
                                    # Normalize DuckDuckGo field names to match Google format
                                    if "link" not in result and "url" in result:
                                        result["link"] = result["url"]
                                    if "title" not in result and "headline" in result:
                                        result["title"] = result["headline"]
                                    if "snippet" not in result and "body" in result:
                                        result["snippet"] = result["body"]
                                    
                                    result["source_query"] = query
                                    result["search_engine"] = "duckduckgo"
                                    result["search_period"] = {
                                        "start": current_date.isoformat(),
                                        "end": window_end.isoformat(),
                                    }
                                    result["page_number"] = page + 1
                                    result["start_offset"] = start_offset
                                
                                ddg_results.extend(page_results)
                                
                                # If we got fewer results than expected, we've reached the end
                                if len(page_results) < 10:
                                    logger.info(f"DuckDuckGo: Reached end of results after page {page + 1}")
                                    break
                                    
                            except Exception as page_error:
                                logger.error(f"Error on DuckDuckGo page {page + 1} for query '{query}': {page_error}")
                                break
                        
                        logger.info(f"DuckDuckGo: Total {len(ddg_results)} results for query '{query}'")
                        all_search_results.extend(ddg_results)

                    except Exception as e:
                        logger.error(f"Error searching DuckDuckGo for query '{query}': {e}")

            # Move to next month
            current_date = window_end
            window_pbar.update(1)

        window_pbar.close()
        
        # Log search engine statistics
        google_results = [r for r in all_search_results if r.get("search_engine") == "google"]
        ddg_results = [r for r in all_search_results if r.get("search_engine") == "duckduckgo"]
        
        logger.info(f"Search engine results:")
        logger.info(f"   🔍 Google: {len(google_results)} results")
        if self.use_duckduckgo:
            logger.info(f"   🦆 DuckDuckGo: {len(ddg_results)} results")
        logger.info(f"   📊 Total before deduplication: {len(all_search_results)}")
        
        return all_search_results

    async def search_and_extract_content(self, search_queries, aliases, parent_company_name, start_date=None, end_date=None):
        """Search for content and extract detailed content with Jina AI + LangChain analysis"""
        
        logger.info(f"Starting fresh search and content extraction")

        # Get search results
        search_results = self.extract_content(search_queries, num_results=self.num_results)
        
        # Deduplicate search results
        search_results = self._deduplicate_search_results(search_results)

        # Extract content and analyze - always fresh
        processed_results = await self.extract_and_analyze_parallel(
            search_results, aliases, parent_company_name
        )

        logger.info(f"Completed search and extraction with {len(processed_results)} articles processed")

        return {
            "search_results": processed_results,
            "timestamp": datetime.now().isoformat(),
        } 