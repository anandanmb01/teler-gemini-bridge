import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Teler Gemini Bridge",
    description="A bridge application between Teler and Gemini Live API for voice calls using media streaming.",
    version="1.0.0"
)

app.add_middleware(
CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    from app.config import settings
    return {
        "message": "Teler Gemini Bridge is running",
        "status": "healthy",
        "server_domain": settings.server_domain,
        "provider": "gemini" if settings.google_api_key else "none"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "teler-gemini-bridge"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
