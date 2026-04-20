"""
api/routes/ml.py — ML label, auto-train, status, samples
"""
from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from core.store import (
    training_samples, ml_model_meta,
    loan_applications, risk_assessments, financial_ratios, entities,
)
from engines.ml_trainer import (
    train, load_meta, model_exists, models_exist,
    MIN_SAMPLES, MIN_PER_CLASS, WEIGHTS,
)
from engines.risk_scoring_engine import reload_model, build_feature_vector, ML_FEATURES

router = APIRouter(prefix="/ml", tags=["ML Training"])


class LabelRequest(BaseModel):
    loan_application_id: str
    label: int          # 0 = healthy, 1 = default
    notes: str = ""


class TrainRequest(BaseModel):
    force: bool = False


def _collect_features(loan_id: str) -> dict:
    loan        = loan_applications.get(loan_id, {})
    entity_id   = loan.get("entity_id", "")
    risk        = risk_assessments.get(loan_id, {})
    ratios_list = financial_ratios.get(entity_id, [])
    latest      = ratios_list[-1] if ratios_list else {}

    research_summary = {
        "negative_count": len(risk.get("risk_signals", [])),
        "has_litigation": any(
            s.get("category") == "litigation_governance"
            for s in risk.get("risk_signals", [])
        ),
    }
    raw = build_feature_vector(latest, research_summary)
    clean = {}
    for k, v in raw.items():
        if isinstance(v, bool): clean[k] = int(v)
        elif v is None:         clean[k] = None
        else:
            try:    clean[k] = float(v)
            except: clean[k] = None
    return clean


def _try_auto_train():
    n_total    = len(training_samples)
    n_defaults = sum(1 for s in training_samples.values() if s.get("label") == 1)
    n_healthy  = n_total - n_defaults
    if n_total >= MIN_SAMPLES and n_defaults >= MIN_PER_CLASS and n_healthy >= MIN_PER_CLASS:
        try:
            logger.info(f"Auto-training with {n_total} samples…")
            meta = train(training_samples)
            ml_model_meta.update(meta)
            reload_model()
            logger.info(f"Auto-train done. Ensemble AUC={meta.get('ensemble_auc')}")
            return meta
        except Exception as e:
            logger.error(f"Auto-train failed: {e}")
    return None


@router.post("/label", status_code=201)
def label_sample(payload: LabelRequest):
    if payload.label not in (0, 1):
        raise HTTPException(400, "label must be 0 (healthy) or 1 (default)")
    loan = loan_applications.get(payload.loan_application_id)
    if not loan:
        raise HTTPException(404, "Loan application not found")
    if loan.get("analysis_status") != "completed":
        raise HTTPException(400, "Analysis must be completed before labeling")

    entity_id   = loan.get("entity_id", "")
    entity      = entities.get(entity_id, {})
    risk        = risk_assessments.get(payload.loan_application_id, {})
    features    = _collect_features(payload.loan_application_id)

    training_samples[payload.loan_application_id] = {
        "loan_id":       payload.loan_application_id,
        "entity_id":     entity_id,
        "company_name":  entity.get("company_name", "Unknown"),
        "label":         payload.label,
        "label_source":  "manual",
        "notes":         payload.notes,
        "labeled_at":    datetime.now().isoformat(),
        "features":      features,
        "risk_score":    risk.get("risk_score"),
        "credit_rating": risk.get("credit_rating"),
    }

    n_total    = len(training_samples)
    n_defaults = sum(1 for s in training_samples.values() if s["label"] == 1)
    n_healthy  = n_total - n_defaults
    auto_meta  = _try_auto_train()

    return {
        "sample_id":      payload.loan_application_id,
        "company_name":   entity.get("company_name"),
        "label":          payload.label,
        "label_text":     "default" if payload.label == 1 else "healthy",
        "total_samples":  n_total,
        "n_defaults":     n_defaults,
        "n_healthy":      n_healthy,
        "ready_to_train": n_total >= MIN_SAMPLES and n_defaults >= MIN_PER_CLASS and n_healthy >= MIN_PER_CLASS,
        "auto_trained":   auto_meta is not None,
        "auto_train_auc": auto_meta.get("ensemble_auc") if auto_meta else None,
        "message": (
            f"Models auto-trained! Ensemble AUC = {auto_meta.get('ensemble_auc')}"
            if auto_meta else
            f"Sample saved. {max(0, MIN_SAMPLES - n_total)} more needed to auto-train."
        ),
    }


@router.post("/train")
def train_model(payload: TrainRequest = TrainRequest()):
    n_total    = len(training_samples)
    n_defaults = sum(1 for s in training_samples.values() if s.get("label") == 1)
    n_healthy  = n_total - n_defaults
    if n_total < MIN_SAMPLES:
        raise HTTPException(400, f"Need >= {MIN_SAMPLES} samples. Have {n_total}.")
    if n_defaults < MIN_PER_CLASS:
        raise HTTPException(400, f"Need >= {MIN_PER_CLASS} default samples. Have {n_defaults}.")
    if n_healthy < MIN_PER_CLASS:
        raise HTTPException(400, f"Need >= {MIN_PER_CLASS} healthy samples. Have {n_healthy}.")
    if models_exist() and not payload.force:
        meta = load_meta()
        if meta.get("n_samples") == n_total:
            raise HTTPException(400, "Already trained on same samples. Pass force=true to retrain.")
    try:
        meta = train(training_samples)
        ml_model_meta.update(meta)
        reload_model()
        return {
            "status":       "trained",
            "n_samples":    meta["n_samples"],
            "n_defaults":   meta["n_defaults"],
            "n_healthy":    meta["n_healthy"],
            "ensemble_auc": meta.get("ensemble_auc"),
            "model_aucs":   meta.get("model_aucs", {}),
            "weights":      meta.get("weights", WEIGHTS),
            "top_features": meta.get("top_features", [])[:5],
            "model_version":meta["model_version"],
            "trained_at":   meta["trained_at"],
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise HTTPException(500, f"Training failed: {e}")


@router.get("/status")
def model_status():
    meta       = load_meta()
    n_total    = len(training_samples)
    n_defaults = sum(1 for s in training_samples.values() if s.get("label") == 1)
    return {
        "models_trained":   meta.get("trained", False),
        "model_version":    meta.get("model_version"),
        "trained_at":       meta.get("trained_at"),
        "ensemble_auc":     meta.get("ensemble_auc"),
        "model_aucs":       meta.get("model_aucs", {}),
        "weights":          meta.get("weights", WEIGHTS),
        "train_samples":    meta.get("n_samples", 0),
        "session_labeled":  n_total,
        "session_defaults": n_defaults,
        "session_healthy":  n_total - n_defaults,
        "ready_to_train":   (n_total >= MIN_SAMPLES and n_defaults >= MIN_PER_CLASS
                             and (n_total - n_defaults) >= MIN_PER_CLASS),
        "min_required":     MIN_SAMPLES,
        "feature_count":    len(ML_FEATURES),
        "top_features":     meta.get("top_features", [])[:12],
    }


@router.get("/samples")
def list_samples():
    return [
        {
            "sample_id":    sid,
            "company_name": s["company_name"],
            "label":        s["label"],
            "label_text":   "default" if s["label"] == 1 else "healthy",
            "risk_score":   s.get("risk_score"),
            "credit_rating":s.get("credit_rating"),
            "labeled_at":   s.get("labeled_at"),
            "notes":        s.get("notes", ""),
        }
        for sid, s in training_samples.items()
    ]


@router.delete("/samples/{sample_id}")
def delete_sample(sample_id: str):
    if sample_id not in training_samples:
        raise HTTPException(404, "Sample not found")
    del training_samples[sample_id]
    return {"deleted": sample_id, "remaining": len(training_samples)}