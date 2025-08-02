import requests

# Both endpoints now on same port
BASE_URL = "http://localhost:8000"

# Step 1: Get aliases
alias_response = requests.post(f"{BASE_URL}/api/aliases/aliases", json={
    "company_name": "Reliance Industries Limited",
    "country": "India"
})

alias_data = alias_response.json()
print(f"✅ Got {len(alias_data['all_aliases'])} aliases")

# Step 2: Search content using aliases
serp_response = requests.post(f"{BASE_URL}/api/serp/search", json={
    "search_queries": alias_data['adverse_search_queries'],
    "aliases": alias_data['all_aliases'],
    "parent_company_name": alias_data['parent_company']
})

serp_data = serp_response.json()
print(serp_data)
#print(f"✅ Found {serp_data['total_articles']} articles")