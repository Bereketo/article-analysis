from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.aliases_endpoint import router as aliases_router
from api.serp_endpoint import router as serp_router
from api.content_extraction_endpoint import router as content_router
from api.article_analysis_endpoint import router as article_router
from api.full_analysis_endpoint import router as full_analysis_router
import os

# Main FastAPI application
app = FastAPI(
    title="Corporate Intelligence API",
    description="""
    A comprehensive API for corporate intelligence tasks including:
    - Company alias generation
    - Web search and content extraction
    - Adverse media screening
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@yourdomain.com"
    },
    license_info={
        "name": "MIT",
    }
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(aliases_router)
app.include_router(serp_router)
app.include_router(content_router)
app.include_router(article_router)
app.include_router(full_analysis_router)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Corporate Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "aliases": "/api/cdd/aliases",
            "aliases_health": "/api/aliases/health",
            "serp_search": "/api/cdd/search", 
            "serp_health": "/api/serp/health",
            "content_extraction": "/api/cdd/extract",
            "content_extraction_health": "/api/content-extraction/health",
            "article_analysis": "/api/cdd/article-analysis",
            "article_analysis_health": "/api/article-analysis/health",
            "full_analysis": "/api/cdd/full-analysis"
        }
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
