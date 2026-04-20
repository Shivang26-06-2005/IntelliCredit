"""
engines/risk_scoring_engine.py

Two-layer scoring:
  Layer 1 — Rule engine (15 rules, always runs)
  Layer 2 — 6-model ensemble + Isolation Forest anomaly score

Final composite = 60% ensemble PD  +  40% rule penalty  (when models exist)
               = BASE + penalties - health_bonuses        (rules-only fallback)
"""
from __future__ import annotations
import pickle
from pathlib import Path
from loguru import logger

_BASE    = Path(__file__).parent.parent / "models"
XGB_PATH = _BASE / "risk_xgb.json"
GBM_PATH = _BASE / "risk_gbm.pkl"
RF_PATH  = _BASE / "risk_rf.pkl"
SVM_PATH = _BASE / "risk_svm.pkl"
LR_PATH  = _BASE / "risk_lr.pkl"
KNN_PATH = _BASE / "risk_knn.pkl"
ISO_PATH = _BASE / "risk_iso.pkl"

WEIGHTS = {"xgb": 0.30, "gbm": 0.25, "rf": 0.20, "svm": 0.10, "lr": 0.10, "knn": 0.05}

RULES: list[dict] = [
    {"id":"R01","description":"Interest coverage below 1",     "key":"interest_coverage",     "op":"<", "threshold":1.0,  "severity":"critical","penalty":30},
    {"id":"R02","description":"DSCR below 1",                  "key":"debt_service_coverage", "op":"<", "threshold":1.0,  "severity":"critical","penalty":25},
    {"id":"R03","description":"Negative net profit margin",    "key":"net_margin_pct",        "op":"<", "threshold":0,    "severity":"high",   "penalty":25},
    {"id":"R04","description":"Current ratio below 1",         "key":"current_ratio",         "op":"<", "threshold":1.0,  "severity":"high",   "penalty":20},
    {"id":"R05","description":"Debt-to-Equity above 4",        "key":"debt_to_equity",        "op":">", "threshold":4.0,  "severity":"high",   "penalty":20},
    {"id":"R06","description":"Negative operating cash flow",  "key":"operating_cf_positive", "op":"==","threshold":False,"severity":"high",   "penalty":15},
    {"id":"R07","description":"Litigation / governance risk",  "key":"has_litigation_signal", "op":"==","threshold":True, "severity":"high",   "penalty":20},
    {"id":"R08","description":"Altman Z in distress zone",     "key":"altman_zone_distress",  "op":"==","threshold":True, "severity":"high",   "penalty":20},
    {"id":"R09","description":"Net Debt/EBITDA above 5",       "key":"net_debt_to_ebitda",    "op":">", "threshold":5.0,  "severity":"high",   "penalty":15},
    {"id":"R10","description":"OCF/Net Profit below 0.5",      "key":"ocf_to_net_profit",     "op":"<", "threshold":0.5,  "severity":"high",   "penalty":15},
    {"id":"R11","description":"Debt-to-Equity above 2",        "key":"debt_to_equity",        "op":">", "threshold":2.0,  "severity":"medium", "penalty":10},
    {"id":"R12","description":"Revenue declining YoY",         "key":"revenue_growth_pct",    "op":"<", "threshold":0,    "severity":"medium", "penalty":10},
    {"id":"R13","description":"Promoter holding below 26%",    "key":"promoter_holding_pct",  "op":"<", "threshold":26,   "severity":"medium", "penalty":10},
    {"id":"R14","description":"Significant negative news",     "key":"negative_news_count",   "op":">", "threshold":3,    "severity":"medium", "penalty":10},
    {"id":"R15","description":"Altman Z in grey zone",         "key":"altman_zone_grey",      "op":"==","threshold":True, "severity":"medium", "penalty":8},
]

HEALTH_BONUSES: list[dict] = [
    {"key":"current_ratio",         "op":">=","threshold":1.5,  "bonus":2},
    {"key":"interest_coverage",     "op":">=","threshold":3.0,  "bonus":3},
    {"key":"debt_service_coverage", "op":">=","threshold":1.5,  "bonus":3},
    {"key":"net_margin_pct",        "op":">=","threshold":10.0, "bonus":2},
    {"key":"debt_to_equity",        "op":"<=","threshold":1.0,  "bonus":2},
    {"key":"revenue_growth_pct",    "op":">=","threshold":5.0,  "bonus":2},
    {"key":"roe",                   "op":">=","threshold":15.0, "bonus":1},
    {"key":"operating_cf_positive", "op":"==","threshold":True, "bonus":3},
    {"key":"altman_zone_safe",      "op":"==","threshold":True, "bonus":4},
    {"key":"ocf_to_net_profit",     "op":">=","threshold":1.0,  "bonus":2},
    {"key":"cash_earnings_quality", "op":">=","threshold":0.8,  "bonus":2},
]

BASE_SCORE = 18.0

ML_FEATURES: list[str] = [
    "current_ratio","quick_ratio","cash_ratio","nwc_to_revenue",
    "debt_to_equity","debt_to_assets","interest_coverage","debt_service_coverage",
    "financial_leverage","lt_debt_ratio","net_debt_to_ebitda","net_debt_to_equity",
    "gross_margin_pct","ebitda_margin_pct","ebit_margin_pct","net_margin_pct",
    "roe","roa","roce","operating_leverage",
    "asset_turnover","equity_turnover","receivables_days","payables_days",
    "inventory_days","cash_conversion_cycle","fixed_asset_turnover","wc_turnover",
    "ocf_to_revenue","ocf_to_net_profit","ocf_to_total_debt","ocf_to_interest",
    "free_cash_flow","capex_to_revenue","cash_earnings_quality",
    "altman_z_score","equity_multiplier",
    "revenue_growth_pct","profit_growth_pct","ebitda_growth_pct",
    "asset_growth_pct","debt_growth_pct",
    "promoter_holding_pct",
    "negative_news_count","has_litigation_flag",
]

# Runtime model cache
_models: dict = {}


def _load_models():
    global _models
    _models = {}
    try:
        import xgboost as xgb
        if XGB_PATH.exists():
            m = xgb.XGBClassifier(); m.load_model(str(XGB_PATH))
            _models["xgb"] = m; logger.info("XGBoost loaded")
    except Exception as e: logger.warning(f"XGBoost load failed: {e}")

    for key, path in [("gbm", GBM_PATH), ("rf", RF_PATH),
                      ("svm", SVM_PATH), ("lr", LR_PATH), ("knn", KNN_PATH)]:
        try:
            if path.exists():
                with open(path, "rb") as f: _models[key] = pickle.load(f)
                logger.info(f"{key.upper()} loaded")
        except Exception as e: logger.warning(f"{key} load failed: {e}")

    try:
        if ISO_PATH.exists():
            with open(ISO_PATH, "rb") as f: _models["iso"] = pickle.load(f)
            logger.info("IsolationForest loaded")
    except Exception as e: logger.warning(f"IsoForest load failed: {e}")


_load_models()


def reload_model():
    _load_models()


def build_feature_vector(ratios: dict, research_summary: dict) -> dict:
    features = {k: ratios.get(k) for k in ML_FEATURES}
    features["negative_news_count"] = research_summary.get("negative_count", 0)
    features["has_litigation_flag"] = 1.0 if research_summary.get("has_litigation", False) else 0.0

    ocf = ratios.get("operating_cash_flow")
    if ocf is not None:
        try:    features["operating_cf_positive"] = float(ocf) > 0
        except: features["operating_cf_positive"] = None
    else:
        ocf_r = ratios.get("ocf_to_revenue")
        features["operating_cf_positive"] = (ocf_r > 0) if ocf_r is not None else None

    zone = ratios.get("altman_zone")
    features["altman_zone_distress"] = (zone == "distress")
    features["altman_zone_grey"]     = (zone == "grey")
    features["altman_zone_safe"]     = (zone == "safe")
    features["has_litigation_signal"] = research_summary.get("has_litigation", False)
    return features


def _to_row(features: dict) -> list:
    row = []
    for k in ML_FEATURES:
        val = features.get(k)
        if isinstance(val, bool): val = 1.0 if val else 0.0
        try:    row.append(float(val) if val is not None else 0.0)
        except: row.append(0.0)
    return row


def evaluate_rules(features: dict) -> tuple[list[dict], float]:
    triggered, total = [], 0.0
    for rule in RULES:
        val = features.get(rule["key"])
        if val is None: continue
        op, thr = rule["op"], rule["threshold"]
        hit = ((op == "<"  and isinstance(val, (int, float)) and val < thr) or
               (op == ">"  and isinstance(val, (int, float)) and val > thr) or
               (op == "==" and val == thr))
        if hit:
            triggered.append({
                "rule_id":     rule["id"],
                "description": rule["description"],
                "severity":    rule["severity"],
                "penalty":     rule["penalty"],
                "actual_value":round(val, 4) if isinstance(val, float) else val,
                "threshold":   thr,
            })
            total += rule["penalty"]
    return triggered, min(total, 100.0)


def _health_bonus(features: dict) -> float:
    bonus = 0.0
    for hb in HEALTH_BONUSES:
        val = features.get(hb["key"])
        if val is None: continue
        op, thr = hb["op"], hb["threshold"]
        if ((op == ">=" and isinstance(val, (int,float)) and val >= thr) or
            (op == "<=" and isinstance(val, (int,float)) and val <= thr) or
            (op == "==" and val == thr)):
            bonus += hb["bonus"]
    return bonus


def compute_risk_score(ratios: dict, research_summary: dict) -> dict:
    features = build_feature_vector(ratios, research_summary)
    triggered_rules, rule_penalty = evaluate_rules(features)

    pd_ensemble   = None
    model_pds     = {}
    anomaly_score = None
    shap_values   = {}

    classifier_keys = [k for k in ["xgb","gbm","rf","svm","lr","knn"] if k in _models]

    if classifier_keys:
        try:
            import pandas as pd
            row = _to_row(features)
            df  = pd.DataFrame([dict(zip(ML_FEATURES, row))])

            raw_pds = {}
            for key in classifier_keys:
                try:
                    raw_pds[key] = float(_models[key].predict_proba(df)[0][1])
                except Exception as e:
                    logger.warning(f"{key} predict failed: {e}")

            if raw_pds:
                w_sum       = sum(WEIGHTS.get(k, 0) for k in raw_pds)
                pd_ensemble = sum(WEIGHTS.get(k, 0) * v for k, v in raw_pds.items()) / w_sum

                model_pds = {
                    "xgboost":            round(raw_pds.get("xgb", 0) * 100, 2) if "xgb" in raw_pds else None,
                    "gradient_boosting":  round(raw_pds.get("gbm", 0) * 100, 2) if "gbm" in raw_pds else None,
                    "random_forest":      round(raw_pds.get("rf",  0) * 100, 2) if "rf"  in raw_pds else None,
                    "svm":                round(raw_pds.get("svm", 0) * 100, 2) if "svm" in raw_pds else None,
                    "logistic_regression":round(raw_pds.get("lr",  0) * 100, 2) if "lr"  in raw_pds else None,
                    "knn":                round(raw_pds.get("knn", 0) * 100, 2) if "knn" in raw_pds else None,
                    "ensemble":           round(pd_ensemble * 100, 2),
                }
                model_pds = {k: v for k, v in model_pds.items() if v is not None}

            # Isolation Forest anomaly score (convert decision_function → 0-100)
            if "iso" in _models:
                import numpy as np
                score_raw = float(_models["iso"].decision_function(df)[0])
                # decision_function: positive = normal, negative = anomaly
                # Map to 0-100 where 100 = extreme anomaly
                anomaly_score = round(max(0.0, min(100.0, (-score_raw + 0.5) * 100)), 1)

            # SHAP on XGBoost
            if "xgb" in _models:
                try:
                    import shap
                    explainer = shap.TreeExplainer(_models["xgb"])
                    sv = explainer.shap_values(df)
                    if isinstance(sv, list): sv = sv[1]
                    shap_values = {f: round(float(v), 6) for f, v in zip(ML_FEATURES, sv[0])}
                except Exception as e:
                    logger.warning(f"SHAP failed: {e}")

        except Exception as e:
            logger.error(f"ML inference failed: {e}")
            pd_ensemble = None

    # Composite score
    if pd_ensemble is not None:
        composite = round((pd_ensemble * 100 * 0.60) + (rule_penalty * 0.40), 2)
        version   = f"ensemble({','.join(classifier_keys)})+rules"
    else:
        composite = round(max(5.0, min(BASE_SCORE + rule_penalty - _health_bonus(features), 100.0)), 2)
        version   = "rules-only"

    rating                        = _score_to_rating(composite)
    recommendation, rationale, conditions = _recommend(composite, triggered_rules)

    return {
        "risk_score":               composite,
        "credit_rating":            rating,
        "probability_of_default":   pd_ensemble,
        "model_pds":                model_pds,
        "anomaly_score":            anomaly_score,
        "feature_vector":           {k: features.get(k) for k in ML_FEATURES},
        "feature_contributions":    shap_values,
        "rule_flags":               triggered_rules,
        "rule_penalty":             rule_penalty,
        "recommendation":           recommendation,
        "recommendation_rationale": rationale,
        "conditions":               conditions,
        "model_version":            version,
    }


RATING_MAP = [(10,"AAA"),(20,"AA"),(30,"A"),(40,"BBB"),(55,"BB"),(70,"B"),(85,"CCC"),(100,"D")]

def _score_to_rating(score: float) -> str:
    for threshold, rating in RATING_MAP:
        if score <= threshold: return rating
    return "D"

def _recommend(score: float, triggered: list) -> tuple[str, str, list]:
    critical = [r for r in triggered if r["severity"] == "critical"]
    high     = [r for r in triggered if r["severity"] == "high"]
    if score <= 35:
        return "approve", "Entity demonstrates strong financial health with manageable risk.", []
    elif score <= 60 and not critical:
        conds = [f"Monitor: {r['description']}" for r in high[:3]]
        if conds: conds.append("Quarterly financial covenant reporting required.")
        return "conditional_approve", "Approval recommended subject to enhanced monitoring.", conds
    else:
        reasons = "; ".join(r["description"] for r in (critical + high)[:5])
        return "reject", f"Rejected due to critical risk indicators: {reasons}", []
