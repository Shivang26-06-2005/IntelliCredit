"""
core/store.py
Simple in-memory store — replaces the entire database.
All data lives in Python dicts for the duration of the server session.
"""
import uuid
from typing import Any

# ── In-memory tables ──────────────────────────────────────────────────────────
entities: dict[str, dict]         = {}
loan_applications: dict[str, dict] = {}
documents: dict[str, dict]        = {}
extracted_data: dict[str, dict]   = {}   # keyed by doc_id
financial_ratios: dict[str, list] = {}   # keyed by entity_id → list of ratio dicts
research_findings: dict[str, list] = {}  # keyed by entity_id
risk_assessments: dict[str, dict] = {}   # keyed by loan_id
credit_reports: dict[str, dict]   = {}   # keyed by loan_id

def new_id() -> str:
    return str(uuid.uuid4())

# ── ML Training Store ─────────────────────────────────────────────────────────
# Each entry: { loan_id, entity_id, features: dict, label: 0|1, labeled_at, company_name }
training_samples: dict[str, dict] = {}

# ML model metadata
ml_model_meta: dict = {
    "trained": False,
    "n_samples": 0,
    "n_defaults": 0,
    "accuracy": None,
    "trained_at": None,
    "feature_names": [],
    "model_version": None,
}
