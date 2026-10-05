"""FastAPI routes for Hybrid RAG."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.pipeline import HybridRAGPipeline, get_pipeline
from app.models.schemas import (
    DocumentInfo, HealthResponse, IngestResponse, IngestTextRequest, QueryRequest, QueryResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(pipeline: HybridRAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    return HealthResponse(status="ok", **pipeline.stats())


@router.post("/ingest", response_model=IngestResponse, tags=["ingest"])
def ingest(req: IngestTextRequest, pipeline: HybridRAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    try:
        return pipeline.ingest_text(req.text, source_id=req.source_id, metadata=req.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ingest/file", response_model=IngestResponse, tags=["ingest"])
async def ingest_file(file: UploadFile = File(...), pipeline: HybridRAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return pipeline.ingest_file(file.filename or "upload", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest, pipeline: HybridRAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    """Retrieve with BM25 **and** dense embeddings, fuse with RRF, answer with citations."""
    try:
        return pipeline.query(req.question, top_k=req.top_k, per_retriever_k=req.per_retriever_k,
                              source_filter=req.source_filter)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents", response_model=list[DocumentInfo], tags=["system"])
def documents(pipeline: HybridRAGPipeline = Depends(get_pipeline)) -> list[DocumentInfo]:
    return pipeline.list_documents()
