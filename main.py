from fastapi import FastAPI
from api.aliases_endpoint import app as aliases_app
from api.serp_endpoint import app as serp_app
from api.content_extraction_endpoint import app as content_extraction_app
from typing import Optional


# Main FastAPI application
app = FastAPI(
    title="Corporate Intelligence API",
    description="API for corporate alias generation and adverse media screening",
    version="1.0.0"
)

# Mount the endpoints
app.mount("/api/aliases", aliases_app)
app.mount("/api/serp", serp_app)
app.mount("/api/content-extraction", content_extraction_app)

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
