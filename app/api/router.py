from fastapi import APIRouter

from app.api.calls import router as calls_router
from app.api.webhooks import router as webhooks_router

router = APIRouter()

router.include_router(calls_router, prefix="/calls", tags=["calls"])
router.include_router(webhooks_router, prefix="/webhooks", tags=["webhooks"])
