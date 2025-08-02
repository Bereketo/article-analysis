import requests
import json

url = "http://localhost:8000/api/serp/search"
payload = {
    "search_queries": ["Reliance Industries fraud", "Reliance Industries scandal"],
    "aliases": ["RIL", "Reliance"],
    "parent_company_name": "Reliance Industries Limited"
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))