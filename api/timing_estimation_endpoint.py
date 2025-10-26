from fastapi import APIRouter

router = APIRouter(
    prefix="/api/cdd",
    tags=["timing"],
    responses={404: {"description": "Not found"}},
)

@router.get("/timing-estimation/health")
async def health_check():
    return {"status": "healthy", "service": "timing-estimation-api"}