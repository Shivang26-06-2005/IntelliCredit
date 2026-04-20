"""
api/routes/documents.py — Document upload and classification (in-memory)
"""
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Optional
from config.settings import get_settings
from core.store import documents, entities, new_id
from core.extraction_engine import classify_document_by_keywords, extract_text_from_file

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED = {".pdf", ".docx", ".doc", ".xlsx", ".csv", ".txt"}

class ClassifyRequest(BaseModel):
    document_type: str
    fiscal_year: Optional[int] = None

# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload/{entity_id}", status_code=201)
async def upload_document(
    entity_id: str,
    fiscal_year: int = Form(None),
    file: UploadFile = File(...),
):
    if entity_id not in entities:
        raise HTTPException(404, "Entity not found")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"File type {ext} not supported")

    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(413, "File too large")

    # Save to disk
    upload_dir = Path(settings.upload_dir) / entity_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    file_path.write_bytes(content)

    # Auto classify
    doc_type, confidence = "unknown", 0.0
    try:
        text = extract_text_from_file(str(file_path), file.content_type or "")
        doc_type, confidence = classify_document_by_keywords(text)
        doc_type = doc_type.value if hasattr(doc_type, 'value') else str(doc_type)
    except Exception:
        pass

    doc = {
        "id": new_id(),
        "entity_id": entity_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "file_size_bytes": len(content),
        "mime_type": file.content_type,
        "document_type": doc_type,
        "classification_confidence": round(confidence, 3),
        "analyst_verified": False,
        "fiscal_year": fiscal_year,
        "status": "classified",
        "created_at": datetime.now().isoformat(),
    }
    documents[doc["id"]] = doc
    return doc

# ── List / Get ────────────────────────────────────────────────────────────────

@router.get("/entity/{entity_id}")
def list_documents(entity_id: str):
    return [d for d in documents.values() if d["entity_id"] == entity_id]

@router.get("/{document_id}")
def get_document(document_id: str):
    doc = documents.get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc

# ── Override classification ───────────────────────────────────────────────────

@router.patch("/{document_id}/classify")
def override_classification(document_id: str, payload: ClassifyRequest):
    doc = documents.get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    doc["document_type"] = payload.document_type
    doc["analyst_verified"] = True
    doc["classification_confidence"] = 1.0
    if payload.fiscal_year:
        doc["fiscal_year"] = payload.fiscal_year
    return doc

# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str):
    doc = documents.pop(document_id, None)
    if not doc:
        raise HTTPException(404, "Document not found")
    try:
        Path(doc["file_path"]).unlink(missing_ok=True)
    except Exception:
        pass
