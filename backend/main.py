"""
Legal Case Intelligence System - FastAPI Backend
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from services.pdf_parser import extract_text_from_pdf
from services.chunker import chunk_legal_text
from services.metadata_extractor import extract_metadata
from services.vector_store import VectorStore
from services.qa_engine import answer_question, summarize_case, find_similar_cases

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data" / "uploads"
EMBED_DIR  = BASE_DIR / "embeddings"
FRONT_DIR  = BASE_DIR / "frontend"
CASES_FILE = BASE_DIR / "data" / "cases.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Legal Case Intelligence System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory=str(FRONT_DIR)), name="static")

# ── singleton vector store ─────────────────────────────────────────────────────
vector_store = VectorStore(embed_dir=str(EMBED_DIR))

# ── helpers ────────────────────────────────────────────────────────────────────
def load_cases() -> dict:
    if CASES_FILE.exists():
        return json.loads(CASES_FILE.read_text())
    return {}

def save_cases(cases: dict):
    CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CASES_FILE.write_text(json.dumps(cases, indent=2, ensure_ascii=False))

# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(FRONT_DIR / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "cases_indexed": len(load_cases())}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, chunk it, embed and index it."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    case_id = str(uuid.uuid4())[:8]
    pdf_path = DATA_DIR / f"{case_id}_{file.filename}"

    # Save file
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract raw text
    try:
        raw_text = extract_text_from_pdf(str(pdf_path))
    except Exception as e:
        raise HTTPException(500, f"PDF parsing failed: {e}")

    if len(raw_text.strip()) < 100:
        raise HTTPException(400, "PDF appears to be empty or image-only (no extractable text)")

    # Extract metadata using LLM
    try:
        metadata = extract_metadata(raw_text[:4000])   # first 4k chars for speed
    except Exception as e:
        metadata = {
            "case_name": file.filename.replace(".pdf", ""),
            "court": "Unknown Court",
            "date": "Unknown",
            "judges": [],
            "parties": {"petitioner": "Unknown", "respondent": "Unknown"},
        }

    # Chunk + embed
    chunks = chunk_legal_text(raw_text, case_id=case_id, metadata=metadata)
    vector_store.add_chunks(chunks)

    # Persist case record
    cases = load_cases()
    cases[case_id] = {
        "id": case_id,
        "filename": file.filename,
        "pdf_path": str(pdf_path),
        "metadata": metadata,
        "chunk_count": len(chunks),
        "raw_text_length": len(raw_text),
    }
    save_cases(cases)

    return {
        "case_id": case_id,
        "filename": file.filename,
        "metadata": metadata,
        "chunks_indexed": len(chunks),
    }


@app.get("/cases")
def list_cases():
    return list(load_cases().values())


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    cases = load_cases()
    if case_id not in cases:
        raise HTTPException(404, "Case not found")
    return cases[case_id]


class QueryRequest(BaseModel):
    question: str
    case_id: Optional[str] = None   # restrict search to one case
    top_k: int = 5


@app.post("/query")
def query(req: QueryRequest):
    """Semantic search + QA over indexed cases."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    results = vector_store.search(req.question, top_k=req.top_k, case_id=req.case_id)
    if not results:
        return {"answer": "No relevant content found. Please upload case PDFs first.", "sources": []}

    answer = answer_question(req.question, results)
    return {"answer": answer, "sources": results}


class SummarizeRequest(BaseModel):
    case_id: str


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    """Generate structured summary for a specific case."""
    cases = load_cases()
    if req.case_id not in cases:
        raise HTTPException(404, "Case not found")

    chunks = vector_store.get_case_chunks(req.case_id)
    if not chunks:
        raise HTTPException(400, "No indexed content for this case")

    summary = summarize_case(chunks, cases[req.case_id]["metadata"])
    return {"case_id": req.case_id, "summary": summary}


class SimilarRequest(BaseModel):
    case_id: str
    top_k: int = 3


@app.post("/similar")
def similar(req: SimilarRequest):
    """Find cases similar to the given case."""
    cases = load_cases()
    if req.case_id not in cases:
        raise HTTPException(404, "Case not found")

    similar_cases = find_similar_cases(req.case_id, vector_store, cases, top_k=req.top_k)
    return {"similar_cases": similar_cases}


@app.delete("/cases/{case_id}")
def delete_case(case_id: str):
    cases = load_cases()
    if case_id not in cases:
        raise HTTPException(404, "Case not found")

    # Remove PDF
    pdf_path = Path(cases[case_id]["pdf_path"])
    if pdf_path.exists():
        pdf_path.unlink()

    # Remove from vector store
    vector_store.remove_case(case_id)

    del cases[case_id]
    save_cases(cases)
    return {"deleted": case_id}
