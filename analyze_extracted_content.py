#!/usr/bin/env python3
"""
Analyze Extracted Content using API Endpoint
============================================

This script reads JSON files containing extracted content and performs
analysis by calling the existing /api/cdd/article-analysis endpoint.

Usage:
    python analyze_extracted_content.py [json_file_path]
    
Examples:
    python analyze_extracted_content.py extracted-content/Adani_Group_content-extraction_20251001_085926.json
    python analyze_extracted_content.py  # Uses latest file from extracted-content/
"""

import asyncio
import json
import os
import sys
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzes extracted content by calling the API endpoint"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.analysis_endpoint = f"{api_base_url}/api/cdd/article-analysis"
    
    def load_extracted_content_file(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load and validate extracted content JSON file
        
        Args:
            json_file_path: Path to the JSON file with extracted content
            
        Returns:
            Dict containing the loaded data
        """
        logger.info(f"📁 Loading extracted content from: {json_file_path}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)
            
            # Validate structure
            if 'metadata' not in extracted_data:
                logger.warning("⚠️ No metadata found in the JSON file")
                
            if 'extracted_content' not in extracted_data:
                logger.error("❌ No extracted_content found in the JSON file")
                return {}
                
            logger.info(f"✅ Successfully loaded extracted content file")
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Error loading JSON file: {e}")
            return {}
    
    def prepare_articles_for_analysis(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert extracted content to the format expected by the API endpoint
        
        Args:
            extracted_data: The loaded extracted content data
            
        Returns:
            List of articles formatted for the API
        """
        extracted_content = extracted_data.get('extracted_content', [])
        articles = []
        
        logger.info(f"📊 Processing {len(extracted_content)} articles from extracted content")
        
        for i, article in enumerate(extracted_content):
            # Check if content was successfully extracted
            jina_content = article.get('jina_content', {})
            content = jina_content.get('content', '')
            
            if not content or content.strip() == '':
                logger.warning(f"⚠️ Skipping article {i+1} with no content: {article.get('link', 'Unknown URL')}")
                continue
            
            # Format article for API
            article_data = {
                "url": article.get('link', ''),
                "content": content,
                "title": article.get('title', jina_content.get('title', '')),
                "search_engine": article.get('search_engine', 'direct'),
                "search_query": article.get('source_query', ''),
                "published_date": jina_content.get('publishedTime', ''),
                "source": article.get('source', '')
            }
            
            articles.append(article_data)
        
        logger.info(f"📋 Prepared {len(articles)} articles with valid content for analysis")
        return articles
    
    def call_analysis_api(self, articles: List[Dict[str, Any]], company_name: str, aliases: List[str]) -> Dict[str, Any]:
        """
        Call the article analysis API endpoint
        
        Args:
            articles: List of articles to analyze
            company_name: Company name (used as parent company)
            aliases: List of company aliases
            
        Returns:
            API response data
        """
        if not articles:
            logger.error("❌ No articles to analyze")
            return {}
        
        # Prepare request payload
        request_payload = {
            "articles": articles,
            "aliases": aliases,
            "parent_company_name": company_name,
            "source": "extracted_content"
        }
        
        logger.info(f"🚀 Calling analysis API with {len(articles)} articles...")
        logger.info(f"🏢 Company: {company_name}")
        logger.info(f"🎯 Aliases: {aliases}")
        logger.info(f"📡 API Endpoint: {self.analysis_endpoint}")
        
        try:
            # Make API request
            response = requests.post(
                self.analysis_endpoint,
                json=request_payload,
                headers={'Content-Type': 'application/json'},
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                logger.info(f"✅ API call successful!")
                return response.json()
            else:
                logger.error(f"❌ API call failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {}
                
        except requests.exceptions.Timeout:
            logger.error("❌ API request timed out (5 minutes)")
            return {}
        except requests.exceptions.ConnectionError:
            logger.error("❌ Could not connect to API server. Make sure the server is running.")
            return {}
        except Exception as e:
            logger.error(f"❌ Error calling API: {e}")
            return {}
    
    def display_analysis_summary(self, api_response: Dict[str, Any]) -> None:
        """
        Display a summary of the analysis results
        
        Args:
            api_response: Response from the analysis API
        """
        if not api_response:
            logger.error("❌ No analysis results to display")
            return
        
        # Extract summary information
        summary = api_response.get('summary', {})
        results = api_response.get('results', [])
        total_articles = api_response.get('total_articles', 0)
        
        print()
        print("📊 ANALYSIS RESULTS SUMMARY")
        print("=" * 50)
        print(f"📋 Total articles processed: {summary.get('total_articles_processed', 0)}")
        print(f"✅ Successful analyses: {summary.get('successful_analyses', 0)}")
        print(f"❌ Failed analyses: {summary.get('failed_analyses', 0)}")
        print(f"🎯 Articles in final results: {total_articles}")
        
        # Risk categories breakdown
        risk_categories = summary.get('risk_categories', {})
        if risk_categories:
            print()
            print("🏷️ Risk Categories:")
            for category, count in risk_categories.items():
                print(f"   • {category}: {count}")
        
        # Adverse analysis breakdown
        if results:
            adverse_count = sum(1 for r in results if r.get('analysis', {}).get('is_adverse') == 'Negative')
            filtered_count = sum(1 for r in results if r.get('analysis', {}).get('is_filter') == True)
            neutral_count = total_articles - adverse_count - filtered_count
            
            print()
            print("📈 Content Analysis Breakdown:")
            print(f"   🚫 Filtered (not relevant): {filtered_count}")
            print(f"   ⚠️  Adverse (negative): {adverse_count}")
            print(f"   😐 Neutral/Positive: {neutral_count}")
            
            if adverse_count > 0:
                print()
                print("⚠️ Adverse Articles Found:")
                adverse_articles = [r for r in results if r.get('analysis', {}).get('is_adverse') == 'Negative']
                for i, article in enumerate(adverse_articles[:5], 1):  # Show first 5
                    title = article.get('title', 'No title')[:80]
                    url = article.get('url', 'No URL')
                    print(f"   {i}. {title}...")
                    print(f"      🔗 {url}")
                
                if len(adverse_articles) > 5:
                    print(f"   ... and {len(adverse_articles) - 5} more adverse articles")
        
        print()
        print(f"🕐 Analysis completed at: {summary.get('analysis_timestamp', 'Unknown')}")


def find_latest_extracted_content_file() -> str:
    """Find the latest extracted content file in the extracted-content directory"""
    
    extracted_content_dir = "extracted-content"
    
    if not os.path.exists(extracted_content_dir):
        logger.error(f"❌ Directory not found: {extracted_content_dir}")
        return ""
    
    # Get all JSON files
    json_files = []
    for filename in os.listdir(extracted_content_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(extracted_content_dir, filename)
            mtime = os.path.getmtime(filepath)
            json_files.append((filepath, mtime))
    
    if not json_files:
        logger.error(f"❌ No JSON files found in {extracted_content_dir}")
        return ""
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: x[1], reverse=True)
    latest_file = json_files[0][0]
    
    logger.info(f"📄 Latest extracted content file: {latest_file}")
    return latest_file


def check_api_server() -> bool:
    """Check if the API server is running"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API server is running")
            return True
        else:
            logger.error(f"❌ API server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to API server at http://localhost:8000")
        logger.error("   Please make sure the server is running with: python main.py")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking API server: {e}")
        return False


def main():
    """Main function with command line argument handling"""
    
    print("=" * 70)
    print("🧠 Content Analysis using API Endpoint")
    print("=" * 70)
    print()
    
    # Check if API server is running
    if not check_api_server():
        print("💡 To start the API server, run:")
        print("   python main.py")
        print()
        sys.exit(1)
    
    # Determine which JSON file to process
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        logger.info(f"📂 Using specified file: {json_file_path}")
    else:
        json_file_path = find_latest_extracted_content_file()
        if not json_file_path:
            logger.error("❌ No JSON file found to process")
            sys.exit(1)
        logger.info(f"📂 Using latest file: {json_file_path}")
    
    # Validate file exists
    if not os.path.exists(json_file_path):
        logger.error(f"❌ File not found: {json_file_path}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = ContentAnalyzer()
    
    # Load extracted content
    extracted_data = analyzer.load_extracted_content_file(json_file_path)
    if not extracted_data:
        logger.error("❌ Failed to load extracted content file")
        sys.exit(1)
    
    # Extract metadata
    metadata = extracted_data.get('metadata', {})
    company_name = metadata.get('company_name', 'Unknown Company')
    aliases = metadata.get('aliases', [company_name])
    
    logger.info(f"🏢 Company: {company_name}")
    logger.info(f"🎯 Aliases: {aliases}")
    
    # Prepare articles for analysis
    articles = analyzer.prepare_articles_for_analysis(extracted_data)
    if not articles:
        logger.error("❌ No articles with valid content found for analysis")
        sys.exit(1)
    
    # Call analysis API
    logger.info("🚀 Starting analysis via API endpoint...")
    start_time = datetime.now()
    
    api_response = analyzer.call_analysis_api(articles, company_name, aliases)
    
    if not api_response:
        logger.error("❌ Analysis failed - no results from API")
        sys.exit(1)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Display results
    analyzer.display_analysis_summary(api_response)
    
    print()
    print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"⏱️ Total time: {duration:.1f} seconds")
    print()
    print("📁 Results have been saved to:")
    print("   • llm-analysis/ directory (JSON format)")
    print("   • Excel files with multiple sheets")
    print("   • PDF adverse media report")
    print("   • Email notification sent (if configured)")
    print()
    print("💡 You can find the generated files in the llm-analysis directory.")


if __name__ == "__main__":
    main()