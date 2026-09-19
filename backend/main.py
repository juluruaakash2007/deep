"""
DeepShield — FastAPI Main Application
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend.config import settings
from backend.database import connect_to_mongo, close_mongo_connection
from backend.routers import auth, detect, history, analytics, report

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger("deepshield")


# ─── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🛡️  DeepShield starting up...")
    await connect_to_mongo()  # initializes JSON file store

    logger.info("HF API mode — inference runs on HuggingFace servers (capcheck/ai-image-detection).")

    # Ensure reports directory exists
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)

    yield

    logger.info("DeepShield shutting down...")
    await close_mongo_connection()


# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepShield API",
    description="Intelligent Synthetic Media Detection Platform",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ─── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(detect.router)
app.include_router(history.router)
app.include_router(analytics.router)
app.include_router(report.router)


# ─── Health Check ───────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status":  "ok",
        "version": settings.APP_VERSION,
    }


# ─── Serve Frontend Static Files ────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(FRONTEND_DIR):
    # /static → serves CSS, JS, images inside frontend/
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/{page}.html", include_in_schema=False)
    async def serve_page(page: str):
        path = os.path.join(FRONTEND_DIR, f"{page}.html")
        if os.path.exists(path):
            return FileResponse(path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ─── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
