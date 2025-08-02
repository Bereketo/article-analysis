from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import re
from datetime import datetime
from urllib.parse import urlparse
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent
import os
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

app = FastAPI()
logger = logging.getLogger(__name__)

def get_llm() -> AzureChatOpenAI:
    """Initialize and return the Azure OpenAI LLM"""
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.1,
        max_tokens=4000
    )

async def format_content_with_llm(text: str, llm: AzureChatOpenAI) -> str:
    """
    Use LLM to clean and format the extracted content for better readability.
    This is more effective than regex for handling complex content structures.
    """
    if not text or len(text.strip()) < 50:  # Skip if text is too short to be meaningful
        return text
        
    try:
        # System message to instruct the LLM how to format the content
        system_message = SystemMessage(content="""
        You are a helpful assistant that cleans and formats web article content.
        Your task is to:
        1. Remove any navigation menus, headers, footers, and ads
        2. Keep only the main article content
        3. Format the text with proper paragraphs and newlines
        4. Ensure each sentence is on a new line when appropriate
        5. Use double newlines between paragraphs
        6. Format lists with proper bullet points and newlines
        7. Preserve tables and format them with proper spacing
        8. Remove any duplicate content
        9. Ensure proper sentence structure
        10. Remove any remaining HTML or markdown tags
        11. Preserve important information like dates, names, and key facts
        
        Important: Make sure to use proper newlines and spacing for better readability.
        Return only the cleaned content, no additional explanations.
        """)
        
        # Prepare the user message with the content to be cleaned
        user_message = HumanMessage(content=f"Clean and format this article content:\n\n{text}")
        
        # Get the response from the LLM
        response = await llm.agenerate(messages=[[system_message, user_message]])
        
        # Extract the cleaned content
        cleaned_content = response.generations[0][0].text.strip()
        
        return cleaned_content if cleaned_content else text  # Fallback to original if empty
        
    except Exception as e:
        logger.error(f"Error formatting content with LLM: {str(e)}")
        return text  # Return original text in case of error

def format_content(text: str) -> str:
    """
    Format the extracted content for better readability by:
    1. Removing HTML tags and scripts
    2. Cleaning up navigation and header/footer content
    3. Removing excessive whitespace and newlines
    4. Ensuring proper paragraph separation
    5. Cleaning up bullet points and lists
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove JavaScript and CSS content
    text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', text, flags=re.DOTALL)
    
    # Remove common navigation patterns (like ========)
    text = re.sub(r'=+\s*', '\n', text)
    
    # Remove empty square brackets (often from removed links)
    text = re.sub(r'\[\s*]', '', text)
    
    # Clean up URLs and email addresses
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)       # Remove email addresses
    
    # Remove common navigation words and phrases
    navigation_phrases = [
        'home', 'about', 'contact', 'login', 'logout', 'sign in', 'sign up',
        'privacy policy', 'terms of service', 'cookie policy', 'advertise',
        'subscribe', 'follow us', 'share', 'menu', 'search', 'categories',
        'trending', 'popular', 'latest', 'more', 'read more', 'continue reading'
    ]
    for phrase in navigation_phrases:
        text = re.sub(rf'\b{re.escape(phrase)}\b', '', text, flags=re.IGNORECASE)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper sentence spacing
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Replace bullet points with proper formatting
    text = re.sub(r'\s*[•●▪]\s*', '\n• ', text)
    
    # Split into paragraphs and clean each one
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    # Filter out very short paragraphs that are likely navigation or ads
    paragraphs = [p for p in paragraphs if len(p.split()) > 3]
    
    # Join paragraphs with double newlines for better readability
    formatted_text = '\n\n'.join(paragraphs)
    
    # Remove any remaining excessive whitespace
    formatted_text = re.sub(r'\s+', ' ', formatted_text).strip()
    
    return formatted_text

# Pydantic models for request/response
class ContentExtractionRequest(BaseModel):
    urls: List[str]
    aliases: List[str]
    parent_company_name: Optional[str] = "Unknown"

class ContentExtractionResponse(BaseModel):
    extracted_content: List[Dict[str, Any]]
    total_articles: int
    processing_summary: Dict[str, Any]

@app.post("/extract", response_model=ContentExtractionResponse)
async def extract_content(content_request: ContentExtractionRequest):
    """
    Extract content from the provided URLs using the content extraction agent.
    This endpoint processes multiple URLs asynchronously and returns the extracted content.
    """
    start_time = datetime.utcnow()
    
    try:
        # Initialize LLM for content formatting
        llm = get_llm()
        
        # Process URLs in batches to avoid overwhelming the system
        extracted_content = []
        failed_extractions = 0
        
        for url in content_request.urls:
            if not url:
                failed_extractions += 1
                continue
                
            try:
                # Extract content using Jina AI
                extractor = ImprovedContentExtractionAgent()
                content = await extractor.extract_content_with_jina(url)
                
                # Get content and metadata
                article_content = content.get("content", "")
                article_title = content.get("title", "")
                article_meta = content.get("meta", {})
                
                # First, clean with regex
                cleaned_content = format_content(article_content)
                
                # Then enhance with LLM for better formatting
                try:
                    formatted_content = await format_content_with_llm(cleaned_content, llm)
                    # Ensure proper newlines in the final output
                    formatted_content = '\n'.join(line.strip() for line in formatted_content.split('\n') if line.strip())
                except Exception as e:
                    logger.error(f"Error in LLM formatting: {str(e)}")
                    formatted_content = cleaned_content
                
                # Create article object
                article = {
                    "url": url,
                    "title": article_title,
                    "content": formatted_content,
                    "metadata": {
                        "source_domain": urlparse(url).netloc,
                        "extracted_at": datetime.utcnow().isoformat(),
                        "content_length": len(formatted_content),
                        "language": article_meta.get("language", "en"),
                        "extraction_status": "success",
                        "extraction_metadata": {"extractor": "jina-ai+llm", "version": "2.0"}
                    }
                }
                
                extracted_content.append(article)
                
            except Exception as e:
                logger.error(f"Error extracting content from {url}: {str(e)}")
                failed_extractions += 1
                
        # Calculate processing time
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Prepare response
        response = ContentExtractionResponse(
            extracted_content=extracted_content,
            total_articles=len(extracted_content),
            processing_summary={
                "total_urls": len(content_request.urls),
                "successful_extractions": len(extracted_content),
                "failed_extractions": failed_extractions,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "parameters": {
                    "aliases_used": content_request.aliases,
                    "parent_company": content_request.parent_company_name
                }
            }
        )
        
        logger.info(f"✅ Content extraction completed: {len(extracted_content)} articles processed")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error during content extraction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to extract content: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "content-extraction-api",
        "agent_type": "jina_content_extractor"
    }
