"""
engines/ml_trainer.py

Trains on ALL analyzed documents — no minimum sample count, no manual labeling.
Label is derived from the rule-based risk score: score >= 55 = default(1), else healthy(0).

Models trained:
  1. XGBoost
  2. GradientBoosting
  3. Random Forest
  4. Logistic Regression
  5. SVM
  6. KNN
  + Isolation Forest (anomaly, unsupervised)
"""
from __future__ import annotations
import json
import pickle
from datetime import datetime
from pathlib import Path
from loguru import logger

MODEL_DIR = Path(__file__).parent.parent / "models"
XGB_PATH  = MODEL_DIR / "risk_xgb.json"
GBM_PATH  = MODEL_DIR / "risk_gbm.pkl"
RF_PATH   = MODEL_DIR / "risk_rf.pkl"
SVM_PATH  = MODEL_DIR / "risk_svm.pkl"
LR_PATH   = MODEL_DIR / "risk_lr.pkl"
KNN_PATH  = MODEL_DIR / "risk_knn.pkl"
ISO_PATH  = MODEL_DIR / "risk_iso.pkl"
META_PATH = MODEL_DIR / "risk_model_meta.json"

# Train after every new document — no minimum needed
MIN_SAMPLES   = 1
MIN_PER_CLASS = 1

WEIGHTS = {"xgb": 0.30, "gbm": 0.25, "rf": 0.20, "svm": 0.10, "lr": 0.10, "knn": 0.05}

FEATURE_NAMES: list[str] = [
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


# Synthetic profiles used ONLY when all real samples are the same class.
# These get replaced automatically once analyst manually labels real data via UI.
SYNTHETIC_DISTRESS = {
    "current_ratio": 0.6, "quick_ratio": 0.3, "cash_ratio": 0.05,
    "debt_to_equity": 7.0, "debt_to_assets": 0.88, "interest_coverage": 0.4,
    "debt_service_coverage": 0.5, "net_margin_pct": -12.0, "ebitda_margin_pct": 1.0,
    "roe": -18.0, "roa": -4.0, "altman_z_score": 0.9, "revenue_growth_pct": -20.0,
    "ocf_to_net_profit": -0.8, "cash_earnings_quality": 0.1, "net_debt_to_ebitda": 12.0,
    "promoter_holding_pct": 15.0, "negative_news_count": 5.0, "has_litigation_flag": 1.0,
}

SYNTHETIC_HEALTHY = {
    "current_ratio": 2.5, "quick_ratio": 1.8, "cash_ratio": 0.7,
    "debt_to_equity": 0.3, "debt_to_assets": 0.25, "interest_coverage": 10.0,
    "debt_service_coverage": 3.0, "net_margin_pct": 18.0, "ebitda_margin_pct": 25.0,
    "roe": 22.0, "roa": 12.0, "altman_z_score": 5.0, "revenue_growth_pct": 15.0,
    "ocf_to_net_profit": 1.4, "cash_earnings_quality": 0.92, "net_debt_to_ebitda": 1.2,
    "promoter_holding_pct": 60.0, "negative_news_count": 0.0, "has_litigation_flag": 0.0,
}


def _build_matrix(training_samples: dict):
    import numpy as np

    labeled = {sid: s for sid, s in training_samples.items() if s.get("label") in (0, 1)}

    def _row(feats):
        row = []
        for fname in FEATURE_NAMES:
            val = feats.get(fname)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            try:
                row.append(float(val) if val is not None else 0.0)
            except (TypeError, ValueError):
                row.append(0.0)
        return row

    X, y = [], []
    for s in labeled.values():
        X.append(_row(s.get("features", {})))
        y.append(int(s["label"]))

    unique = set(y)

    # Check if analyst has manually labeled both classes already
    manual_classes = {s["label"] for s in labeled.values() if s.get("label_source") == "manual"}

    if len(unique) < 2 and len(manual_classes) < 2:
        # Only one class in real data — add synthetic opposite to enable training
        # This is temporary; replaced once analyst manually labels real data of both classes
        if unique == {0} or len(unique) == 0:
            logger.info("Only healthy samples — adding synthetic distress for initial training. "
                        "Use the Label panel in UI to add real default examples.")
            X.append(_row(SYNTHETIC_DISTRESS))
            y.append(1)
        else:
            logger.info("Only default samples — adding synthetic healthy for initial training. "
                        "Use the Label panel in UI to add real healthy examples.")
            X.append(_row(SYNTHETIC_HEALTHY))
            y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labeled


def _cv_auc(estimator, X, y, cv):
    from sklearn.model_selection import cross_val_score
    try:
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")
        return round(float(scores.mean()), 4)
    except Exception as e:
        logger.warning(f"CV failed ({type(estimator).__name__}): {e}")
        return None


def train(training_samples: dict) -> dict:
    try:
        import numpy as np
        import xgboost as xgb
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold
        from sklearn.dummy import DummyClassifier
    except ImportError as e:
        raise RuntimeError(f"Missing: {e}. Run: pip install xgboost scikit-learn numpy")

    X, y, labeled = _build_matrix(training_samples)
    n = len(labeled)
    n_defaults = int(y.sum())
    n_healthy  = n - n_defaults

    if n == 0:
        raise ValueError("No labeled samples found.")

    logger.info(f"Training on {n} samples ({n_defaults} defaults, {n_healthy} healthy)")

    scale_pos   = max(n_healthy, 1) / max(n_defaults, 1)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Only do CV when we have enough samples of each class
    can_cv      = n >= 6 and n_defaults >= 2 and n_healthy >= 2
    n_splits    = min(3, n_defaults, n_healthy) if can_cv else 0
    cv          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if n_splits >= 2 else None

    results: dict[str, dict] = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_m = xgb.XGBClassifier(
        n_estimators=max(50, min(300, n * 10)),
        max_depth=min(4, max(2, n // 3)),
        learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=1,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    xgb_auc = _cv_auc(xgb_m, X, y, cv) if cv else None
    xgb_m.fit(X, y)
    xgb_m.save_model(str(XGB_PATH))
    xgb_imp = dict(zip(FEATURE_NAMES, [round(float(v), 6) for v in xgb_m.feature_importances_]))
    results["xgb"] = {"cv_auc": xgb_auc, "importance": xgb_imp}
    logger.info(f"XGBoost trained. AUC={xgb_auc}")

    # ── Gradient Boosting ─────────────────────────────────────────────────────
    gbm_m = GradientBoostingClassifier(
        n_estimators=max(50, min(200, n * 10)),
        max_depth=min(3, max(2, n // 3)),
        learning_rate=0.1, subsample=0.8,
        min_samples_leaf=1, random_state=42,
    )
    gbm_auc = _cv_auc(gbm_m, X, y, cv) if cv else None
    gbm_m.fit(X, y)
    with open(GBM_PATH, "wb") as f: pickle.dump(gbm_m, f)
    gbm_imp = dict(zip(FEATURE_NAMES, [round(float(v), 6) for v in gbm_m.feature_importances_]))
    results["gbm"] = {"cv_auc": gbm_auc, "importance": gbm_imp}
    logger.info(f"GBM trained. AUC={gbm_auc}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_m = RandomForestClassifier(
        n_estimators=max(50, min(300, n * 10)),
        max_depth=min(6, max(2, n // 2)),
        min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_auc = _cv_auc(rf_m, X, y, cv) if cv else None
    rf_m.fit(X, y)
    with open(RF_PATH, "wb") as f: pickle.dump(rf_m, f)
    rf_imp = dict(zip(FEATURE_NAMES, [round(float(v), 6) for v in rf_m.feature_importances_]))
    results["rf"] = {"cv_auc": rf_auc, "importance": rf_imp}
    logger.info(f"RF trained. AUC={rf_auc}")

    # ── SVM ───────────────────────────────────────────────────────────────────
    svm_pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced", probability=True, random_state=42)),
    ])
    svm_auc = _cv_auc(svm_pipe, X, y, cv) if cv else None
    svm_pipe.fit(X, y)
    with open(SVM_PATH, "wb") as f: pickle.dump(svm_pipe, f)
    results["svm"] = {"cv_auc": svm_auc, "importance": {}}
    logger.info(f"SVM trained. AUC={svm_auc}")

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                  max_iter=2000, solver="lbfgs", random_state=42)),
    ])
    lr_auc = _cv_auc(lr_pipe, X, y, cv) if cv else None
    lr_pipe.fit(X, y)
    with open(LR_PATH, "wb") as f: pickle.dump(lr_pipe, f)
    lr_coefs = dict(zip(FEATURE_NAMES,
        [round(abs(float(c)), 6) for c in lr_pipe.named_steps["lr"].coef_[0]]))
    results["lr"] = {"cv_auc": lr_auc, "importance": lr_coefs}
    logger.info(f"LR trained. AUC={lr_auc}")

    # ── KNN ───────────────────────────────────────────────────────────────────
    k = max(1, min(5, n - 1))
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance",
                                     metric="euclidean", n_jobs=-1)),
    ])
    knn_auc = _cv_auc(knn_pipe, X, y, cv) if cv else None
    knn_pipe.fit(X, y)
    with open(KNN_PATH, "wb") as f: pickle.dump(knn_pipe, f)
    results["knn"] = {"cv_auc": knn_auc, "importance": {}}
    logger.info(f"KNN trained. AUC={knn_auc}")

    # ── Isolation Forest ──────────────────────────────────────────────────────
    contamination = min(0.45, max(0.01, n_defaults / max(n, 1)))
    iso_m = IsolationForest(n_estimators=100, contamination=contamination,
                            random_state=42, n_jobs=-1)
    iso_m.fit(X)
    with open(ISO_PATH, "wb") as f: pickle.dump(iso_m, f)
    logger.info("IsoForest trained.")

    # Ensemble AUC
    valid   = {k: results[k]["cv_auc"] for k in results if results[k].get("cv_auc") is not None}
    w_sum   = sum(WEIGHTS[k] for k in valid) if valid else 1
    ensemble_auc = round(sum(WEIGHTS[k] * valid[k] for k in valid) / w_sum, 4) if valid else None

    top_features = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:12]

    meta = {
        "trained":         True,
        "n_samples":       n,
        "n_defaults":      n_defaults,
        "n_healthy":       n_healthy,
        "ensemble_auc":    ensemble_auc,
        "model_aucs":      {k: results[k]["cv_auc"] for k in results},
        "weights":         WEIGHTS,
        "trained_at":      datetime.now().isoformat(),
        "feature_names":   FEATURE_NAMES,
        "feature_count":   len(FEATURE_NAMES),
        "top_features":    [{"feature": f, "importance": i} for f, i in top_features],
        "xgb_importance":  xgb_imp,
        "rf_importance":   rf_imp,
        "lr_coefficients": lr_coefs,
        "model_version":   f"ensemble6-{datetime.now().strftime('%Y%m%d-%H%M')}",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info(f"All models trained. Ensemble AUC={ensemble_auc}. Samples={n}")
    return meta


def load_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text())
        except Exception:
            pass
    return {"trained": False, "n_samples": 0}


def models_exist() -> bool:
    return all(p.exists() for p in [XGB_PATH, GBM_PATH, RF_PATH, SVM_PATH, LR_PATH, KNN_PATH])


def model_exists() -> bool:
    return XGB_PATH.exists()