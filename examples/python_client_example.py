import requests
import json

# Updated API endpoints (both running on port 8000)
ALIASES_API_URL = "http://localhost:8000/api/aliases/aliases"
SERP_API_URL = "http://localhost:8000/api/serp/search"

def search_company_content(search_queries, aliases, parent_company_name="Unknown"):
    """
    Search for company content using SERP endpoint
    """
    payload = {
        "search_queries": search_queries,
        "aliases": aliases,
        "parent_company_name": parent_company_name
    }
    
    try:
        response = requests.post(SERP_API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling SERP API: {e}")
        return None

# Example usage
adani_queries = [
    "Adani Enterprises adverse news",
    "Adani Enterprises fraud allegations",
    "Adani Enterprises regulatory issues"
]

adani_aliases = [
    "Adani Enterprises Limited",
    "AEL",
    "Adani Group"
]

print("🔍 Searching for Adani Enterprises content...")
results = search_company_content(
    search_queries=adani_queries,
    aliases=adani_aliases,
    parent_company_name="Adani Group"
)

if results:
    print(f"✅ Found {results['total_articles']} articles")