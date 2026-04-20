"""
core/analysis_orchestrator.py
"""
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from loguru import logger

from config.settings import get_settings
from agents.financial_agent import compute_ratios
from agents.research_agent import ResearchAgent
from agents.report_agent import ReportAgent
from core.extraction_engine import (
    apply_schema_mapping, extract_financials_with_llm,
    extract_tables_from_pdf, extract_text_from_file,
)
from engines.risk_scoring_engine import compute_risk_score, build_feature_vector
from core.store import (
    entities, loan_applications, documents,
    extracted_data, financial_ratios,
    research_findings, risk_assessments, credit_reports,
    training_samples, ml_model_meta,
)

settings = get_settings()


def _label_from_financials(ratios: dict, financials: dict) -> tuple[int, str]:
    """
    Derive default/healthy label purely from financial data — NOT from rule score.

    A company is labeled DEFAULT (1) if it meets ANY critical distress signal:
      - Net profit is negative (loss-making)
      - Interest coverage < 1.0 (cannot cover interest from operations)
      - Current ratio < 1.0 (cannot meet short-term obligations)
      - Debt-to-equity > 4.0 (severely over-leveraged)
      - Altman Z < 1.81 (distress zone)
      - Operating cash flow is negative

    Otherwise HEALTHY (0).

    This mirrors how credit analysts actually classify companies —
    from the financial statements, not from a scoring model.
    """
    reasons = []

    net_margin  = ratios.get("net_margin_pct")
    int_cov     = ratios.get("interest_coverage")
    curr_ratio  = ratios.get("current_ratio")
    de_ratio    = ratios.get("debt_to_equity")
    altman_z    = ratios.get("altman_z_score")
    ocf         = financials.get("operating_cash_flow")
    net_profit  = financials.get("net_profit")

    if net_margin is not None and net_margin < 0:
        reasons.append("negative net margin")
    if net_profit is not None:
        try:
            if float(net_profit) < 0:
                reasons.append("net loss")
        except (TypeError, ValueError):
            pass
    if int_cov is not None and int_cov < 1.0:
        reasons.append(f"interest coverage {int_cov:.2f} < 1")
    if curr_ratio is not None and curr_ratio < 1.0:
        reasons.append(f"current ratio {curr_ratio:.2f} < 1")
    if de_ratio is not None and de_ratio > 4.0:
        reasons.append(f"D/E {de_ratio:.2f} > 4")
    if altman_z is not None and altman_z < 1.81:
        reasons.append(f"Altman Z {altman_z:.2f} in distress zone")
    if ocf is not None:
        try:
            if float(ocf) < 0:
                reasons.append("negative OCF")
        except (TypeError, ValueError):
            pass

    if reasons:
        return 1, f"DEFAULT — {'; '.join(reasons)}"
    return 0, "HEALTHY — no critical distress signals"


class AnalysisOrchestrator:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.report_agent   = ReportAgent()

    async def run_full_analysis(self, loan_id: str) -> dict:
        logger.info(f"Starting analysis: {loan_id}")

        loan   = loan_applications.get(loan_id)
        entity = entities.get(loan["entity_id"]) if loan else None
        if not loan or not entity:
            raise ValueError("Loan or entity not found")

        loan["analysis_status"] = "running"

        try:
            # 1. Extract
            financials_by_year = self._process_documents(entity["id"])

            # 2. Ratios
            ratios_by_year    = self._compute_ratios(entity["id"], financials_by_year)
            latest_ratios     = _latest(ratios_by_year)
            latest_financials = _latest(financials_by_year)

            # 3. Research
            findings = await self.research_agent.run(
                company_name=entity["company_name"],
                sector=entity.get("sector"),
            )
            research_findings[entity["id"]] = findings
            research_summary = _summarize_research(findings)

            # 4. Score (rule-based first pass)
            ratios_for_scoring = {**latest_ratios}
            if latest_financials.get("operating_cash_flow") is not None:
                ratios_for_scoring["operating_cash_flow"] = latest_financials["operating_cash_flow"]

            risk = compute_risk_score(ratios_for_scoring, research_summary)
            risk["risk_signals"] = research_summary.get("signals", [])

            # 5. Label from raw financials — independent of rule score
            label, label_reason = _label_from_financials(latest_ratios, latest_financials)
            logger.info(f"Auto-label for {entity.get('company_name')}: {label_reason}")

            # 6. Save training sample with financially-derived label
            self._add_training_sample(
                loan_id=loan_id,
                entity=entity,
                ratios=latest_ratios,
                research_summary=research_summary,
                label=label,
                label_reason=label_reason,
                risk_score=risk["risk_score"],
                credit_rating=risk.get("credit_rating"),
            )

            # 7. Train/retrain all ML models on accumulated samples
            self._train_models()

            # 8. Re-score — if ML trained, this now uses ensemble PD
            from engines.risk_scoring_engine import reload_model
            reload_model()
            risk = compute_risk_score(ratios_for_scoring, research_summary)
            risk["risk_signals"]        = research_summary.get("signals", [])
            risk["loan_application_id"] = loan_id
            risk["entity_id"]           = entity["id"]
            risk["label_reason"]        = label_reason
            risk_assessments[loan_id]   = risk

            # 9. SWOT
            swot = self.report_agent.generate_swot(
                entity=entity,
                ratios=latest_ratios,
                risk_flags=risk.get("rule_flags", []),
                research_findings=findings,
            )
            risk["swot"] = swot
            risk_assessments[loan_id] = risk

            # 10. CAM + PDF
            cam_md = self.report_agent.generate_cam_report(
                entity=entity, loan_application=loan,
                financials=latest_financials, ratios=latest_ratios,
                risk_assessment=risk, research_findings=findings, swot=swot,
            )
            pdf_path = self.report_agent.export_to_pdf(
                cam_md, _pdf_path(entity["company_name"], loan_id), entity["company_name"],
            )
            credit_reports[loan_id] = {
                "loan_application_id": loan_id,
                "entity_id":           entity["id"],
                "report_content":      cam_md,
                "report_path":         pdf_path,
            }

            loan["analysis_status"] = "completed"
            logger.info(
                f"Done: {loan_id} | {risk.get('credit_rating')} | "
                f"score={risk.get('risk_score')} | {risk.get('model_version')}"
            )
            return risk

        except Exception as e:
            loan["analysis_status"] = "failed"
            logger.exception(f"Analysis failed: {e}")
            raise

    def _add_training_sample(self, loan_id, entity, ratios, research_summary,
                              label, label_reason, risk_score, credit_rating):
        try:
            features = build_feature_vector(ratios, research_summary)
            clean = {}
            for k, v in features.items():
                if isinstance(v, bool):
                    clean[k] = int(v)
                elif v is None:
                    clean[k] = None
                else:
                    try:    clean[k] = float(v)
                    except: clean[k] = None

            training_samples[loan_id] = {
                "loan_id":       loan_id,
                "entity_id":     entity["id"],
                "company_name":  entity.get("company_name", "Unknown"),
                "label":         label,
                "label_source":  "financial_ratios",
                "label_reason":  label_reason,
                "labeled_at":    datetime.now().isoformat(),
                "features":      clean,
                "risk_score":    risk_score,
                "credit_rating": credit_rating,
            }
            logger.info(
                f"Sample saved: {entity.get('company_name')} → {label_reason} | "
                f"Total: {len(training_samples)}"
            )
        except Exception as e:
            logger.error(f"Failed to add training sample: {e}")

    def _train_models(self):
        from engines.ml_trainer import train
        n = len(training_samples)
        if n == 0:
            return
        try:
            logger.info(f"Training ML on {n} sample(s)...")
            meta = train(training_samples)
            ml_model_meta.update(meta)
            logger.info(f"ML trained: {n} samples | AUC={meta.get('ensemble_auc')}")
        except Exception as e:
            logger.error(f"ML training failed: {e}")

    def _process_documents(self, entity_id: str) -> dict[int, dict]:
        docs = [d for d in documents.values() if d["entity_id"] == entity_id]
        financials_by_year: dict[int, dict] = {}
        for doc in docs:
            try:
                text   = extract_text_from_file(doc["file_path"], doc.get("mime_type") or "")
                tables = extract_tables_from_pdf(doc["file_path"]) if doc["file_path"].endswith(".pdf") else []
                raw    = extract_financials_with_llm(text)
                normalized = apply_schema_mapping(raw)
                normalized["entity_id"]        = entity_id
                normalized["raw_tables"]        = tables[:5]
                normalized["extraction_method"] = "ollama"
                raw_fy   = normalized.get("fiscal_year") or doc.get("fiscal_year") or "0"
                fy_match = re.search(r"(20\d{2})", str(raw_fy))
                fy       = int(fy_match.group(1)) if fy_match else 0
                extracted_data[doc["id"]] = normalized
                financials_by_year[fy]    = normalized
            except Exception as e:
                logger.error(f"Doc {doc['id']} failed: {e}")
        return financials_by_year

    def _compute_ratios(self, entity_id: str, financials_by_year: dict) -> dict[int, dict]:
        years = sorted(financials_by_year.keys())
        ratios_by_year: dict[int, dict] = {}
        for i, year in enumerate(years):
            prior = financials_by_year.get(years[i - 1]) if i > 0 else None
            r = compute_ratios(financials_by_year[year], prior)
            r["entity_id"]    = entity_id
            ratios_by_year[year] = r
        financial_ratios[entity_id] = list(ratios_by_year.values())
        return ratios_by_year


def _latest(by_year: dict) -> dict:
    return by_year[max(by_year.keys())] if by_year else {}

def _summarize_research(findings: list) -> dict:
    negative = [f for f in findings if f.get("sentiment") == "negative"]
    signals  = []
    for f in negative:
        topics = f.get("key_topics", [])
        if any(t in topics for t in ["litigation", "governance"]):
            signals.append({"category": "litigation_governance", "headline": f.get("headline")})
        elif "debt" in topics:
            signals.append({"category": "debt_stress", "headline": f.get("headline")})
    return {
        "negative_count": len(negative),
        "has_litigation":  any(
            "litigation" in (f.get("key_topics") or []) or
            "governance" in (f.get("key_topics") or [])
            for f in findings
        ),
        "avg_sentiment":  sum(f.get("sentiment_score", 0) for f in findings) / len(findings) if findings else 0.0,
        "signals":        signals[:10],
    }

def _pdf_path(company_name: str, loan_id: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in company_name)[:30]
    reports_dir = Path(settings.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return str(reports_dir / f"CAM_{safe}_{loan_id[:8]}.pdf")