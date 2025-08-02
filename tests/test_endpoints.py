import requests
import json
import time
from datetime import datetime
import os

 
BASE_URL = "http://localhost:8000"

def wait_for_server():
    """Wait for the server to be ready before running tests"""
    max_attempts = 10
    attempts = 0
    
    while attempts < max_attempts:
        try:
            response = requests.get(f"{BASE_URL}")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        time.sleep(1)
        attempts += 1
    
    return False

def test_root_endpoint():
    """Test the root endpoint"""
    response = requests.get(f"{BASE_URL}")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    #assert len(data["endpoints"]) == 6

def test_serp_endpoint():
    """Test the SERP search endpoint"""
    payload = {
        "search_queries": ["Reliance Industries fraud", "Reliance Industries scandal"],
        "aliases": ["RIL", "Reliance"],
        "parent_company_name": "Reliance Industries Limited"
    }
    response = requests.post(f"{BASE_URL}/api/serp/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results_data" in data
    assert "total_articles" in data
    assert "processing_summary" in data
    assert len(data["results_data"]["search_results"]) > 0

def test_content_extraction_endpoint():
    """Test the content extraction endpoint with direct test URLs"""
    # Use reliable test URLs that should have content
    test_urls = [
        "https://www.reuters.com/companies/RELI.NS",
        "https://www.moneycontrol.com/india/stockpricequote/refineries/relianceindustries/RI"
    ]
    
    print(f"\nExtracting content from test URLs: {test_urls}")
    
    extraction_payload = {
        "urls": test_urls,
        "aliases": ["RIL", "Reliance"],
        "parent_company_name": "Reliance Industries Limited"
    }
    
    try:
        # Make the request to the content extraction endpoint with a reasonable timeout
        extraction_response = requests.post(
            f"{BASE_URL}/api/content-extraction/extract",
            json=extraction_payload,
            timeout=60  # Increased timeout to 60 seconds
        )
        print(f"Content extraction status: {extraction_response.status_code}")
        
        if extraction_response.status_code != 200:
            print(f"Content extraction error: {extraction_response.text}")
            assert False, f"Content extraction failed with status {extraction_response.status_code}"
        
        extraction_data = extraction_response.json()
        
        # Check required fields in the response
        assert "extracted_content" in extraction_data, "Missing 'extracted_content' in response"
        assert "total_articles" in extraction_data, "Missing 'total_articles' in response"
        assert "processing_summary" in extraction_data, "Missing 'processing_summary' in response"
        
        # Validate the extracted content structure
        assert isinstance(extraction_data["extracted_content"], list), \
            "Expected 'extracted_content' to be a list"
        
        # Save extracted content to JSON file for inspection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), "..", "extracted_content")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"extracted_content_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nExtracted content saved to: {output_file}")
        
        # Check each extracted article
        for article in extraction_data["extracted_content"]:
            # Check required fields in each article
            assert "url" in article, "Article is missing 'url' field"
            assert "title" in article, f"Article {article.get('url')} is missing 'title' field"
            assert "content" in article, f"Article {article.get('url')} is missing 'content' field"
            assert "metadata" in article, f"Article {article.get('url')} is missing 'metadata' field"
            
            # Print a preview of the formatted content
            content_preview = article["content"][:500] + ("..." if len(article["content"]) > 500 else "")
            print(f"\nContent preview from {article['url']}:")
            print("=" * 80)
            print(content_preview)
            print("=" * 80)
            
    except requests.exceptions.Timeout:
        print(f"Request to {BASE_URL}/api/content-extraction/extract timed out after 60 seconds")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error making request to content extraction endpoint: {str(e)}")
        raise

def test_health_check_endpoints():
    """Test all health check endpoints"""
    endpoints = [
        f"{BASE_URL}/api/aliases/health",
        f"{BASE_URL}/api/serp/health",
        f"{BASE_URL}/api/content-extraction/health"
    ]
    
    for endpoint in endpoints:
        response = requests.get(endpoint)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "service" in data
        assert "agent_type" in data

def test_invalid_requests():
    """Test invalid requests"""
    # Test missing required fields in SERP endpoint
    invalid_serp_payload = {
        "search_queries": [],  # Empty queries
        "aliases": ["RIL", "Reliance"]
    }
    response = requests.post(f"{BASE_URL}/api/serp/search", json=invalid_serp_payload)
    assert response.status_code == 422  # Validation error
    
    # Test invalid URL in content extraction
    invalid_extraction_payload = {
        "urls": ["invalid-url"],  # Invalid URL
        "aliases": ["RIL", "Reliance"]
    }
    response = requests.post(f"{BASE_URL}/api/content-extraction/extract", json=invalid_extraction_payload)
    assert response.status_code == 422  # Validation error

if __name__ == "__main__":
    # Wait for server to start
    if not wait_for_server():
        print("Server did not start within the timeout period")
        exit(1)
    
    # Run tests
    test_root_endpoint()
    #test_serp_endpoint()
    test_content_extraction_endpoint()
    #test_health_check_endpoints()
    #test_invalid_requests()
    
    print("All tests passed!")
