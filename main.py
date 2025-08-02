from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from api.aliases_endpoint import app as aliases_app
from api.serp_endpoint import app as serp_app
from api.content_extraction_endpoint import app as content_extraction_app
from typing import Optional, List, Dict, Any
import os

# Main FastAPI application
app = FastAPI(
    title="Corporate Intelligence API",
    description="""
    A comprehensive API for corporate intelligence tasks including:
    - Company alias generation
    - Web search and content extraction
    - Adverse media screening
    
    ### Available Endpoints
    - `/api/aliases/*`: Generate company name aliases and variations
    - `/api/serp/*`: Perform web searches and extract search results
    - `/api/content-extraction/*`: Extract and process content from web pages
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@yourdomain.com"
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "aliases",
            "description": "Operations with company name aliases and variations"
        },
        {
            "name": "serp",
            "description": "Search Engine Results Page (SERP) operations"
        },
        {
            "name": "content-extraction",
            "description": "Web content extraction and processing"
        }
    ]
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the endpoints BEFORE defining custom_openapi
app.mount("/api/aliases", aliases_app)
app.mount("/api/serp", serp_app)
app.mount("/api/content-extraction", content_extraction_app)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Corporate Intelligence API",
        version="1.0.0",
        description="""
        A comprehensive API for corporate intelligence tasks including:
        - Company alias generation
        - Web search and content extraction
        - Adverse media screening
        """,
        routes=app.routes,
    )
    
    # Add server URL for production
    server_url = os.getenv("SERVER_URL", "https://article-analysis-0e4x.onrender.com")
    openapi_schema["servers"] = [{"url": server_url}]
    
    # Merge OpenAPI schemas from mounted apps
    for route in app.routes:
        if hasattr(route, 'app') and hasattr(route.app, 'openapi'):
            sub_openapi = route.app.openapi()
            if sub_openapi and 'paths' in sub_openapi:
                # Get the mount path properly
                mount_path = route.path.rstrip('/')
                
                for path, methods in sub_openapi['paths'].items():
                    # Combine mount path with sub-app path
                    full_path = f"{mount_path}{path}"
                    openapi_schema['paths'][full_path] = methods
                
                # Merge components (schemas, etc.)
                if 'components' in sub_openapi:
                    if 'components' not in openapi_schema:
                        openapi_schema['components'] = {}
                    for component_type, components in sub_openapi['components'].items():
                        if component_type not in openapi_schema['components']:
                            openapi_schema['components'][component_type] = {}
                        openapi_schema['components'][component_type].update(components)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Corporate Intelligence API",
        "endpoints": {
            "aliases": "/api/aliases/aliases",
            "aliases_health": "/api/aliases/health",
            "serp_search": "/api/serp/search", 
            "serp_health": "/api/serp/health",
            "content_extraction": "/api/content-extraction/extract",
            "content_extraction_health": "/api/content-extraction/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
