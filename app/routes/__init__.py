from fastapi import APIRouter

from app.routes.calls import router as calls_router
from app.routes.webhooks import router as webhooks_router

router = APIRouter()
router.include_router(calls_router, prefix="/calls", tags=["calls"])
router.include_router(webhooks_router, prefix="/webhooks", tags=["webhooks"])

__all__ = ["router"]
