"""
main.py - IntelliCredit (No-DB, in-memory version)
Run: uvicorn main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from config.settings import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.reports_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"IntelliCredit started — model: {settings.ollama_model}")
    yield
    logger.info("IntelliCredit shutting down.")

app = FastAPI(
    title="IntelliCredit API",
    description="AI-Powered Corporate Credit Intelligence (In-Memory)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import entities, documents, analysis, ml
app.include_router(entities.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(ml.router,       prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.ollama_model}

@app.get("/")
async def root():
    return {"message": "IntelliCredit API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
