"""
agents/report_agent.py — SWOT + CAM report generation + PDF export
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from config.settings import get_settings
from utils.llm_client import get_llm_client

settings = get_settings()

# ─── SWOT ─────────────────────────────────────────────────────────────────────

SWOT_SYSTEM = "You are a senior credit analyst. Analyze the financial data and return a SWOT analysis as JSON. No explanation. No markdown."

SWOT_PROMPT = """Analyze this company and return a credit risk SWOT as JSON.

Company: {company_name} | Sector: {sector} | Revenue: Rs {annual_turnover} Cr
Key Ratios: {ratios_summary}
Risk Flags: {risk_flags}

Return ONLY this JSON with 2 specific points each based on the data above:
{{"strengths":["<specific strength 1>","<specific strength 2>"],"weaknesses":["<specific weakness 1>","<specific weakness 2>"],"opportunities":["<specific opportunity 1>","<specific opportunity 2>"],"threats":["<specific threat 1>","<specific threat 2>"]}}"""

# ─── CAM ──────────────────────────────────────────────────────────────────────

CAM_SYSTEM = """You are a senior credit analyst at a leading Indian bank writing for the credit committee.
Your CAM must be thorough, analytical, and banker-grade — not generic.
Every section must reference specific numbers from the data provided.
Use INR Crores. Format in clean Markdown."""

CAM_PROMPT = """Write a complete, professional Credit Appraisal Memo (CAM) for the loan application below.
Every section must cite specific numbers. Be analytical, not descriptive.

═══════════════════════════════════════════════
LOAN APPLICATION
═══════════════════════════════════════════════
Borrower        : {company_name}
CIN             : {cin}
Sector          : {sector}
Loan Type       : {loan_type}
Proposed Amount : Rs {loan_amount} Cr
Tenure          : {tenure_months} months
Interest Rate   : {interest_rate}% p.a.
Purpose         : {loan_purpose}

═══════════════════════════════════════════════
FINANCIAL SUMMARY (FY {fiscal_year})
═══════════════════════════════════════════════
Revenue               : Rs {revenue} Cr
EBITDA                : Rs {ebitda} Cr  (Margin: {ebitda_margin}%)
EBIT                  : Rs {ebit} Cr
Net Profit            : Rs {net_profit} Cr  (Margin: {net_margin}%)
Total Assets          : Rs {total_assets} Cr
Total Debt            : Rs {total_debt} Cr
Net Worth             : Rs {net_worth} Cr
Operating Cash Flow   : Rs {operating_cf} Cr
Free Cash Flow        : Rs {free_cash_flow} Cr
Capex                 : Rs {capex} Cr

═══════════════════════════════════════════════
RATIO DASHBOARD
═══════════════════════════════════════════════
LIQUIDITY
  Current Ratio       : {current_ratio}
  Quick Ratio         : {quick_ratio}
  Cash Ratio          : {cash_ratio}
  NWC / Revenue       : {nwc_to_revenue}%

LEVERAGE
  Debt / Equity       : {debt_to_equity}x
  Debt / Assets       : {debt_to_assets}x
  Net Debt / EBITDA   : {net_debt_to_ebitda}x
  Interest Coverage   : {interest_coverage}x
  DSCR                : {dscr}x
  Financial Leverage  : {financial_leverage}x

PROFITABILITY
  Gross Margin        : {gross_margin}%
  EBITDA Margin       : {ebitda_margin}%
  Net Margin          : {net_margin}%
  ROE                 : {roe}%
  ROA                 : {roa}%
  ROCE                : {roce}%

EFFICIENCY
  Asset Turnover      : {asset_turnover}x
  Receivables Days    : {receivables_days} days
  Payables Days       : {payables_days} days
  Inventory Days      : {inventory_days} days
  Cash Conv. Cycle    : {ccc} days

CASH FLOW QUALITY
  OCF / Net Profit    : {ocf_to_net_profit}x
  OCF / Total Debt    : {ocf_to_total_debt}x
  Cash Earnings Qual. : {cash_earnings_quality}x

GROWTH (YoY)
  Revenue Growth      : {revenue_growth}%
  Profit Growth       : {profit_growth}%
  EBITDA Growth       : {ebitda_growth}%
  Debt Growth         : {debt_growth}%

DISTRESS INDICATOR
  Altman Z-Score      : {altman_z} ({altman_zone} zone)
  Promoter Holding    : {promoter_pct}%

═══════════════════════════════════════════════
RISK ASSESSMENT
═══════════════════════════════════════════════
Risk Score       : {risk_score}/100
Credit Rating    : {credit_rating}
Prob. of Default : {pod}%
ML Model         : {model_version}
Recommendation   : {recommendation}

RULE FLAGS:
{rule_flags}

RISK SIGNALS:
{risk_signals}

═══════════════════════════════════════════════
SWOT
═══════════════════════════════════════════════
Strengths    : {strengths}
Weaknesses   : {weaknesses}
Opportunities: {opportunities}
Threats      : {threats}

═══════════════════════════════════════════════
EXTERNAL RESEARCH
═══════════════════════════════════════════════
{research_findings}

═══════════════════════════════════════════════

Write the full CAM with ALL of the following sections.
Each section must contain specific numbers — no vague statements.

# Credit Appraisal Memo — {company_name}

## 1. Executive Summary
(3-4 sentences: loan ask, key financials, rating, recommendation)

## 2. Borrower Profile
(Business description, sector position, key products/services, years in operation)

## 3. Industry & Market Analysis
(Sector outlook, competitive positioning, tailwinds/headwinds, regulatory environment)

## 4. Financial Performance Analysis
### 4.1 Income Statement Analysis
(Revenue trend, margin analysis, EBITDA quality, profit sustainability)
### 4.2 Balance Sheet Strength
(Asset quality, working capital position, debt structure, equity base)
### 4.3 Cash Flow Analysis
(OCF adequacy, FCF generation, capex cycle, cash earnings quality assessment)

## 5. Ratio Analysis & Benchmarking
(Comment on each ratio group — liquidity, leverage, profitability, efficiency — vs industry norms)

## 6. Debt Serviceability
(Repayment capacity analysis: DSCR, interest coverage, projected EMI vs OCF)

## 7. Altman Z-Score Analysis
(Interpret the score, implications for default risk, trend if prior year available)

## 8. Management & Governance
(Promoter commitment, shareholding stability, governance flags from research)

## 9. Risk Analysis
(Summarise all rule flags and research signals with specific mitigation commentary)

## 10. SWOT Summary
(Brief narrative version of the SWOT, tied to credit decision)

## 11. Credit Rating Rationale
(Explain the {credit_rating} rating — what drives it, what could change it)

## 12. Recommendation & Conditions
(Clear approve/conditional/reject with specific covenant conditions and monitoring requirements)

## 13. Conclusion
(One paragraph summary for credit committee)"""


class ReportAgent:
    def __init__(self):
        self.llm = get_llm_client()

    # ─── SWOT ─────────────────────────────────────────────────────────────────

    def generate_swot(self, entity, ratios, risk_flags, research_findings) -> dict:
        default = {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}
        if not self.llm.is_available():
            logger.warning("LLM not available. Returning empty SWOT.")
            return default

        def _fmt(k):
            v = ratios.get(k)
            if v is None:
                return "N/A"
            return f"{v:.2f}" if isinstance(v, float) else str(v)

        # Keep ratios_summary short — fits within 768 token budget
        key_ratios = ["current_ratio","debt_to_equity","interest_coverage","net_margin_pct","roe","revenue_growth_pct"]
        ratios_summary = ", ".join(
            f"{k.replace('_',' ')}: {_fmt(k)}" for k in key_ratios if ratios.get(k) is not None
        ) or "N/A"

        risk_flags_text = "; ".join(r["description"] for r in risk_flags[:3]) or "None"

        prompt = SWOT_PROMPT.format(
            company_name=entity.get("company_name", "N/A"),
            sector=entity.get("sector", "N/A"),
            annual_turnover=entity.get("annual_turnover", "N/A"),
            ratios_summary=ratios_summary,
            risk_flags=risk_flags_text,
        )
        try:
            result = self.llm.chat_json(prompt, system=SWOT_SYSTEM, temperature=0.1)
            # Validate — reject if model returned template placeholders literally
            if (isinstance(result, dict) and
                "strengths" in result and
                isinstance(result.get("strengths"), list) and
                len(result.get("strengths", [])) > 0 and
                result["strengths"][0] not in ("point1", "<specific strength 1>", "...", "")):
                return result
            logger.warning(f"SWOT returned invalid/template content: {result}")
            return default
        except Exception as e:
            logger.error(f"SWOT generation failed: {e}")
            return default

    # ─── CAM ──────────────────────────────────────────────────────────────────

    def generate_cam_report(self, entity, loan_application, financials,
                            ratios, risk_assessment, research_findings, swot) -> str:
        if not self.llm.is_available():
            return "# Credit Appraisal Memo\n\n**Error:** LLM is not available."

        def _to_float(val):
            if val is None: return None
            try: return float(str(val).replace(",","").strip())
            except: return None

        def _f(d, k, default="N/A"):
            v = d.get(k)
            if v is None:
                return default
            try:
                return f"{float(str(v).replace(',','')):,.2f}"
            except (TypeError, ValueError):
                return str(v)

        def _r(k, default="N/A"):
            return _f(ratios, k, default)

        # Derive ebitda/ebit from ratios if raw financials has null (common after LLM extraction)
        revenue_f = _to_float(financials.get("revenue"))
        ebitda_val = (financials.get("ebitda") or
                      (round(revenue_f * (_to_float(ratios.get("ebitda_margin_pct")) or 0) / 100, 2)
                       if revenue_f else None))
        pbt_f      = _to_float(financials.get("pbt"))
        int_f      = _to_float(financials.get("interest_expense"))
        ebit_val   = (financials.get("ebit") or
                      (round(pbt_f + int_f, 2) if pbt_f is not None and int_f is not None else None))

        pod = risk_assessment.get("probability_of_default")
        pod_str = f"{pod*100:.1f}" if pod is not None else "N/A"

        rule_flags_text = "\n".join(
            f"  [{r.get('severity','?').upper()}] {r.get('description','')} "
            f"(Actual: {r.get('actual_value')}, Threshold: {r.get('threshold')}, Penalty: -{r.get('penalty')}pts)"
            for r in risk_assessment.get("rule_flags", [])
        ) or "  None triggered."

        risk_signals_text = "\n".join(
            f"  [{s.get('category','?').upper().replace('_',' ')}] {s.get('headline','')}"
            for s in risk_assessment.get("risk_signals", [])[:6]
        ) or "  No material signals."

        research_text = "\n".join(
            f"  [{f.get('sentiment','?').upper()}] {f.get('headline','')} — {(f.get('summary') or '')[:120]}"
            for f in research_findings[:8]
        ) or "  No external research available."

        prompt = CAM_PROMPT.format(
            company_name   = entity.get("company_name", "N/A"),
            cin            = entity.get("cin", "N/A"),
            sector         = entity.get("sector", "N/A"),
            loan_type      = loan_application.get("loan_type", "N/A"),
            loan_amount    = _f(loan_application, "loan_amount"),
            tenure_months  = loan_application.get("tenure_months", "N/A"),
            interest_rate  = _f(loan_application, "interest_rate"),
            loan_purpose   = loan_application.get("loan_purpose", "N/A"),
            fiscal_year    = financials.get("fiscal_year", "N/A"),
            revenue        = _f(financials, "revenue"),
            ebitda         = _f({"v": ebitda_val}, "v") if ebitda_val else "N/A",
            ebit           = _f({"v": ebit_val},   "v") if ebit_val   else "N/A",
            net_profit     = _f(financials, "net_profit"),
            total_assets   = _f(financials, "total_assets"),
            total_debt     = _f(financials, "total_debt"),
            net_worth      = _f(financials, "shareholders_equity"),
            operating_cf   = _f(financials, "operating_cash_flow"),
            free_cash_flow = _r("free_cash_flow"),
            capex          = _f(financials, "capex"),
            # Ratios
            current_ratio        = _r("current_ratio"),
            quick_ratio          = _r("quick_ratio"),
            cash_ratio           = _r("cash_ratio"),
            nwc_to_revenue       = _r("nwc_to_revenue"),
            debt_to_equity       = _r("debt_to_equity"),
            debt_to_assets       = _r("debt_to_assets"),
            net_debt_to_ebitda   = _r("net_debt_to_ebitda"),
            interest_coverage    = _r("interest_coverage"),
            dscr                 = _r("debt_service_coverage"),
            financial_leverage   = _r("financial_leverage"),
            gross_margin         = _r("gross_margin_pct"),
            ebitda_margin        = _r("ebitda_margin_pct"),
            net_margin           = _r("net_margin_pct"),
            roe                  = _r("roe"),
            roa                  = _r("roa"),
            roce                 = _r("roce"),
            asset_turnover       = _r("asset_turnover"),
            receivables_days     = _r("receivables_days"),
            payables_days        = _r("payables_days"),
            inventory_days       = _r("inventory_days"),
            ccc                  = _r("cash_conversion_cycle"),
            ocf_to_net_profit    = _r("ocf_to_net_profit"),
            ocf_to_total_debt    = _r("ocf_to_total_debt"),
            cash_earnings_quality= _r("cash_earnings_quality"),
            revenue_growth       = _r("revenue_growth_pct"),
            profit_growth        = _r("profit_growth_pct"),
            ebitda_growth        = _r("ebitda_growth_pct"),
            debt_growth          = _r("debt_growth_pct"),
            altman_z             = _r("altman_z_score"),
            altman_zone          = ratios.get("altman_zone", "N/A"),
            promoter_pct         = _r("promoter_holding_pct"),
            risk_score           = _f(risk_assessment, "risk_score"),
            credit_rating        = risk_assessment.get("credit_rating", "N/A"),
            pod                  = pod_str,
            model_version        = risk_assessment.get("model_version", "rules-only"),
            recommendation       = str(risk_assessment.get("recommendation", "N/A")).replace("_"," ").upper(),
            rule_flags           = rule_flags_text,
            risk_signals         = risk_signals_text,
            strengths            = "; ".join(swot.get("strengths", [])) or "N/A",
            weaknesses           = "; ".join(swot.get("weaknesses", [])) or "N/A",
            opportunities        = "; ".join(swot.get("opportunities", [])) or "N/A",
            threats              = "; ".join(swot.get("threats", [])) or "N/A",
            research_findings    = research_text,
        )

        try:
            report = self.llm.chat(prompt, system=CAM_SYSTEM, temperature=0.15)
            logger.info(f"CAM generated ({len(report)} chars)")
            return report
        except Exception as e:
            logger.error(f"CAM generation failed: {e}")
            return f"# Credit Appraisal Memo\n\n**Error:** {e}"

    # ─── PDF ──────────────────────────────────────────────────────────────────

    def export_to_pdf(self, cam_markdown: str, output_path: str, company_name: str) -> str:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import cm
            from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            doc = SimpleDocTemplate(
                output_path, pagesize=A4,
                leftMargin=2*cm, rightMargin=2*cm,
                topMargin=2.5*cm, bottomMargin=2.5*cm,
            )
            styles = getSampleStyleSheet()
            NAVY  = colors.HexColor("#0f2545")
            BLUE  = colors.HexColor("#1a56db")
            LGRAY = colors.HexColor("#f7f8fa")

            title_s = ParagraphStyle("T",  parent=styles["Title"],   fontSize=20, textColor=NAVY, spaceAfter=4)
            meta_s  = ParagraphStyle("M",  parent=styles["Normal"],  fontSize=8,  textColor=colors.grey, spaceAfter=14)
            h1_s    = ParagraphStyle("H1", parent=styles["Heading1"],fontSize=13, textColor=NAVY, spaceBefore=14, spaceAfter=4)
            h2_s    = ParagraphStyle("H2", parent=styles["Heading2"],fontSize=11, textColor=BLUE, spaceBefore=10, spaceAfter=3)
            h3_s    = ParagraphStyle("H3", parent=styles["Heading3"],fontSize=10, textColor=NAVY, spaceBefore=6,  spaceAfter=2)
            body_s  = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=9,  leading=14, spaceAfter=4)
            code_s  = ParagraphStyle("C",  parent=styles["Code"],    fontSize=8,  leading=12, spaceAfter=3,
                                     fontName="Courier", textColor=colors.HexColor("#333333"))
            bullet_s= ParagraphStyle("BL", parent=styles["Normal"],  fontSize=9,  leading=13,
                                     leftIndent=12, spaceAfter=3)

            story = []
            story.append(Paragraph("Credit Appraisal Memo", title_s))
            story.append(Paragraph(f"Borrower: <b>{company_name}</b>", meta_s))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M IST')}", meta_s))
            story.append(HRFlowable(width="100%", thickness=2, color=NAVY))
            story.append(Spacer(1, 10))

            for line in cam_markdown.split("\n"):
                line = line.rstrip()
                if not line:
                    story.append(Spacer(1, 4))
                elif line.startswith("# "):
                    story.append(Paragraph(line[2:], h1_s))
                    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
                elif line.startswith("## "):
                    story.append(Paragraph(line[3:], h2_s))
                elif line.startswith("### "):
                    story.append(Paragraph(f"<b>{line[4:]}</b>", h3_s))
                elif line.startswith(("- ", "* ", "• ")):
                    story.append(Paragraph(f"• {line[2:]}", bullet_s))
                elif line.startswith("|"):
                    story.append(Paragraph(line.replace("|", "  |  "), code_s))
                elif line.startswith("**") and line.endswith("**"):
                    story.append(Paragraph(f"<b>{line[2:-2]}</b>", body_s))
                elif "═" in line or "─" in line:
                    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dee2e6")))
                else:
                    # Bold inline **text**
                    safe = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
                    story.append(Paragraph(safe, body_s))

            doc.build(story)
            logger.info(f"PDF saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            txt = output_path.replace(".pdf", ".md")
            Path(txt).write_text(cam_markdown)
            return txt