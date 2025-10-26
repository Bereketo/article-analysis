from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.aliases_endpoint import router as aliases_router
from api.serp_endpoint import router as serp_router
from api.content_extraction_endpoint import router as content_router
from api.article_analysis_endpoint import router as article_router
from api.full_analysis_endpoint import router as aliases2
from api.timing_estimation_endpoint import router as timing_router
from api.full_analysis_with_db import router as db_router
import os
import logging
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Global flag to ensure logging is only configured once
_logging_configured = False

# Configure logging for the entire application with proper handler management
def setup_logging():
    """Setup centralized logging configuration to prevent handler conflicts"""
    global _logging_configured
    
    # Only configure logging once to prevent conflicts
    if _logging_configured:
        return logging.getLogger(__name__)
    
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to prevent conflicts
    for handler in root_logger.handlers[:]:
        try:
            root_logger.removeHandler(handler)
            handler.close()
        except Exception:
            pass  # Ignore errors when closing handlers
    
    # Set root logger level
    root_logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and configure console handler with proper exception handling
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    except Exception as e:
        print(f"Warning: Could not setup console logging: {e}")
    
    # Create and configure file handler ONLY if not in reload mode
    # File handlers cause "I/O operation on closed file" errors in reload mode
    is_reload_mode = os.getenv("UVICORN_RELOAD", "false").lower() == "true" or '--reload' in sys.argv
    
    if not is_reload_mode:
        try:
            file_handler = logging.FileHandler('app.log', encoding='utf-8', mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print("✅ File logging enabled: app.log")
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    else:
        print("ℹ️  File logging disabled in reload mode to prevent I/O errors")
    
    # Set specific logger levels
    logging.getLogger("agents.improved_content_extraction_agent").setLevel(logging.INFO)
    logging.getLogger("api.content_extraction_endpoint").setLevel(logging.INFO)
    logging.getLogger("api.serp_endpoint").setLevel(logging.INFO)
    logging.getLogger("api.article_analysis_endpoint").setLevel(logging.INFO)
    logging.getLogger("api.full_analysis_endpoint").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Reduce uvicorn noise
    
    # Prevent reconfiguration
    _logging_configured = True
    
    return logging.getLogger(__name__)

# Setup logging
logger = setup_logging()
logger.info("🚀 Starting Corporate Intelligence API with centralized logging")

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

# Startup event to verify logging
@app.on_event("startup")
async def startup_event():
    logger.info("🔥 FastAPI Application Starting Up")
    logger.info("📊 Logging Configuration:")
    logger.info(f"   📋 Root logger level: {logging.getLogger().level}")
    logger.info(f"   🔍 Agent logger level: {logging.getLogger('agents.improved_content_extraction_agent').level}")
    logger.info(f"   🌐 API logger level: {logging.getLogger('api.content_extraction_endpoint').level}")
    logger.info("✅ All loggers configured successfully")

# Shutdown event to properly close logging handlers
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 FastAPI Application Shutting Down")
    
    # Properly close all logging handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass  # Ignore errors during shutdown

# Include the routers
app.include_router(aliases_router)
app.include_router(serp_router)
app.include_router(content_router)
app.include_router(article_router)
app.include_router(aliases2)
app.include_router(timing_router)
app.include_router(db_router)

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
            "timing_estimation": "/api/cdd/timing-estimation",
            "timing_estimation_health": "/api/cdd/timing-estimation/health",
            "aliases2": "/api/cdd/aliases2",
            "create_report": "/api/cdd/create-report",
            "continue_analysis": "/api/cdd/continue-with-tracking",
            "report_status": "/api/cdd/report-status/{uuid}",
            "list_reports": "/api/cdd/reports",
            "article_grouping_test": "/api/cdd/article-grouping/test"
        }
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
