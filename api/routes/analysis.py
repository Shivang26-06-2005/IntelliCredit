"""
api/routes/analysis.py — Analysis trigger and results (in-memory)
"""
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from core.store import (
    entities, loan_applications, documents,
    financial_ratios, research_findings,
    risk_assessments, credit_reports
)
from core.analysis_orchestrator import AnalysisOrchestrator

router = APIRouter(prefix="/analysis", tags=["Analysis"])

class TriggerRequest(BaseModel):
    loan_application_id: str
    force_refresh: bool = False

# ── Trigger ───────────────────────────────────────────────────────────────────

@router.post("/trigger", status_code=202)
async def trigger_analysis(payload: TriggerRequest, background_tasks: BackgroundTasks):
    loan = loan_applications.get(payload.loan_application_id)
    if not loan:
        raise HTTPException(404, "Loan application not found")
    background_tasks.add_task(_run, payload.loan_application_id)
    return {"loan_application_id": payload.loan_application_id,
            "status": "running", "message": "Analysis started. Poll /status for updates."}

@router.get("/status/{loan_id}")
def get_status(loan_id: str):
    loan = loan_applications.get(loan_id)
    if not loan:
        raise HTTPException(404, "Loan not found")
    return {"loan_application_id": loan_id,
            "status": loan.get("analysis_status", "pending")}

# ── Results ───────────────────────────────────────────────────────────────────

@router.get("/risk/{loan_id}")
def get_risk(loan_id: str):
    r = risk_assessments.get(loan_id)
    if not r:
        raise HTTPException(404, "Risk assessment not found. Run analysis first.")
    return r

@router.get("/ratios/{entity_id}")
def get_ratios(entity_id: str):
    return financial_ratios.get(entity_id, [])

@router.get("/research/{entity_id}")
def get_research(entity_id: str):
    return research_findings.get(entity_id, [])

@router.get("/report/{loan_id}/content")
def get_report_content(loan_id: str):
    r = credit_reports.get(loan_id)
    if not r:
        raise HTTPException(404, "Report not found. Run analysis first.")
    return {"content": r.get("report_content", "")}

@router.get("/report/{loan_id}/download")
def download_report(loan_id: str):
    r = credit_reports.get(loan_id)
    if not r or not r.get("report_path"):
        raise HTTPException(404, "Report not found. Run analysis first.")
    path = Path(r["report_path"])
    if not path.exists():
        raise HTTPException(404, "Report file missing from disk.")
    return FileResponse(
        path=str(path),
        media_type="application/pdf" if path.suffix == ".pdf" else "text/markdown",
        filename=path.name,
    )

# ── Background runner ─────────────────────────────────────────────────────────

async def _run(loan_id: str):
    try:
        orchestrator = AnalysisOrchestrator()
        await orchestrator.run_full_analysis(loan_id)
    except Exception as e:
        from loguru import logger
        logger.exception(f"Analysis failed: {e}")
        if loan_id in loan_applications:
            loan_applications[loan_id]["analysis_status"] = "failed"
