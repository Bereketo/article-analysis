# Updated curl examples (port 8000)

# SERP search
curl -X POST "http://localhost:8000/api/serp/search" \
  -H "Content-Type: application/json" \
  -d '{
    "search_queries": [
      "Adani Enterprises adverse news",
      "Adani Enterprises fraud",
      "Adani Enterprises controversy"
    ],
    "aliases": ["Adani Enterprises Limited", "AEL", "Adani Group"],
    "parent_company_name": "Adani Group"
  }'

# Health checks
curl -X GET "http://localhost:8000/api/serp/health"
curl -X GET "http://localhost:8000/api/aliases/health"

# Root endpoint to see all available endpoints
curl -X GET "http://localhost:8000/"