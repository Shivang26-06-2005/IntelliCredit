"""
api/routes/entities.py — Entity and Loan Application endpoints (in-memory)
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.store import entities, loan_applications, new_id

router = APIRouter(prefix="/entities", tags=["Entities & Loans"])

# ── Schemas ───────────────────────────────────────────────────────────────────

class EntityCreate(BaseModel):
    company_name: str
    cin: Optional[str] = None
    pan: Optional[str] = None
    sector: Optional[str] = None
    subsector: Optional[str] = None
    annual_turnover: Optional[float] = None
    headquarters: Optional[str] = None

class LoanCreate(BaseModel):
    entity_id: str
    loan_type: str
    loan_amount: float
    tenure_months: int
    interest_rate: Optional[float] = None
    loan_purpose: Optional[str] = None

# ── Entity endpoints ──────────────────────────────────────────────────────────

@router.post("/", status_code=201)
def create_entity(payload: EntityCreate):
    eid = new_id()
    entity = {"id": eid, "created_at": datetime.now().isoformat(), **payload.model_dump()}
    entities[eid] = entity
    return entity

@router.get("/")
def list_entities():
    return list(entities.values())

@router.get("/{entity_id}")
def get_entity(entity_id: str):
    e = entities.get(entity_id)
    if not e:
        raise HTTPException(404, "Entity not found")
    return e

# ── Loan endpoints ────────────────────────────────────────────────────────────

@router.post("/{entity_id}/loans", status_code=201)
def create_loan(entity_id: str, payload: LoanCreate):
    if entity_id not in entities:
        raise HTTPException(404, "Entity not found")
    lid = new_id()
    loan = {
        "id": lid,
        "entity_id": entity_id,
        "analysis_status": "pending",
        "created_at": datetime.now().isoformat(),
        **payload.model_dump()
    }
    loan_applications[lid] = loan
    return loan

@router.get("/{entity_id}/loans")
def list_loans(entity_id: str):
    return [l for l in loan_applications.values() if l["entity_id"] == entity_id]

@router.get("/loans/{loan_id}")
def get_loan(loan_id: str):
    loan = loan_applications.get(loan_id)
    if not loan:
        raise HTTPException(404, "Loan not found")
    return loan
