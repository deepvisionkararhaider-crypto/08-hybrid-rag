"""FastAPI application entrypoint for Hybrid RAG."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.pipeline import get_pipeline

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    logger.info("Hybrid RAG service started.")
    yield
    logger.info("Hybrid RAG service shutting down.")


app = FastAPI(
    title="Hybrid RAG",
    version=__version__,
    description=(
        "Retrieval that combines lexical BM25 with dense semantic embeddings and "
        "fuses their rankings with Reciprocal Rank Fusion. Runs fully offline with a "
        "deterministic embedder, numpy store and extractive generator; upgrades to "
        "transformer embeddings + FAISS + a real LLM via environment variables."
    ),
    lifespan=lifespan,
)

_origins = settings.cors_origin_list or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials="*" not in _origins,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {"service": "hybrid-rag", "version": __version__, "health": "/health",
            "docs": "/docs", "openapi": "/openapi.json"}


if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(str(_FRONTEND_DIR / "index.html"))


def main() -> None:
    uvicorn.run("app.main:app", host=settings.host, port=int(os.getenv("PORT", settings.port)), reload=False)


if __name__ == "__main__":
    main()
