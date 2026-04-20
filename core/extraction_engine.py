"""
core/extraction_engine.py - Document extraction (no DB, plain strings for doc types)
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any
from loguru import logger
from config.settings import get_settings
from utils.llm_client import get_llm_client

settings = get_settings()

# ── Document type constants (plain strings) ───────────────────────────────────
ANNUAL_REPORT        = "annual_report"
ALM_STATEMENT        = "alm_statement"
SHAREHOLDING_PATTERN = "shareholding_pattern"
BORROWING_PROFILE    = "borrowing_profile"
CASH_FLOW            = "cash_flow"
PNL_STATEMENT        = "pnl_statement"
BALANCE_SHEET        = "balance_sheet"
PORTFOLIO_PERFORMANCE= "portfolio_performance"
UNKNOWN              = "unknown"

CLASSIFICATION_KEYWORDS: dict[str, list[str]] = {
    ANNUAL_REPORT:        ["annual report", "directors' report", "auditors' report", "standalone financial", "consolidated financial", "notes to accounts"],
    ALM_STATEMENT:        ["asset liability", "alm", "maturity profile", "liquidity gap", "interest rate sensitivity"],
    SHAREHOLDING_PATTERN: ["shareholding pattern", "promoter holding", "public shareholding", "category of shareholders", "foreign institutional"],
    BORROWING_PROFILE:    ["borrowing profile", "debt schedule", "loan repayment", "term loan outstanding", "credit facilities"],
    CASH_FLOW:            ["cash flow", "operating activities", "investing activities", "financing activities"],
    PNL_STATEMENT:        ["profit and loss", "statement of profit", "income statement", "revenue from operations", "total expenses"],
    BALANCE_SHEET:        ["balance sheet", "statement of financial position", "total assets", "total liabilities", "shareholders equity"],
}

def classify_document_by_keywords(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for doc_type, keywords in CLASSIFICATION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score:
            scores[doc_type] = score
    if not scores:
        return UNKNOWN, 0.0
    best = max(scores, key=scores.__getitem__)
    confidence = min(scores[best] / len(CLASSIFICATION_KEYWORDS[best]), 1.0)
    return best, round(confidence, 3)

# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
    if not text.strip():
        try:
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            logger.error(f"pymupdf failed: {e}")
    return text

def extract_text_from_file(file_path: str, mime_type: str = "") -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf" or "pdf" in mime_type:
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        from docx import Document
        return "\n".join(p.text for p in Document(file_path).paragraphs)
    elif ext in (".txt", ".csv"):
        return Path(file_path).read_text(errors="replace")
    return extract_text_from_pdf(file_path)

def extract_tables_from_pdf(file_path: str) -> list[dict]:
    tables = []
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                for t_idx, raw in enumerate(page.extract_tables() or []):
                    if not raw or len(raw) < 2:
                        continue
                    headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(raw[0])]
                    rows = [{headers[i]: str(c).strip() if c else "" for i, c in enumerate(row)}
                            for row in raw[1:] if any(c for c in row)]
                    tables.append({"page": page_num, "table_index": t_idx, "headers": headers, "rows": rows})
    except Exception as e:
        logger.error(f"Table extraction failed: {e}")
    return tables

# ── LLM Financial Extraction ──────────────────────────────────────────────────

EXTRACTION_SYSTEM = "You are a financial data extraction specialist. Return ONLY valid JSON. No explanation. No markdown. No extra text."

# Two small prompts so each response fits within 512 num_predict tokens.
# Prompt A: income statement fields (14 keys)
EXTRACTION_PROMPT_A = '''Extract from the text. Return ONLY this JSON (null if not found):
{{"fiscal_year":null,"currency_unit":"crores","revenue":null,"other_income":null,"total_income":null,"cost_of_goods_sold":null,"gross_profit":null,"interest_expense":null,"pbt":null,"tax":null,"net_profit":null,"depreciation":null}}
Text: {text}'''

# Prompt B: balance sheet + cash flow fields (22 keys)
EXTRACTION_PROMPT_B = '''Extract from the text. Return ONLY this JSON (null if not found):
{{"ebitda":null,"ebit":null,"total_assets":null,"current_assets":null,"cash_and_equivalents":null,"inventories":null,"trade_receivables":null,"total_liabilities":null,"current_liabilities":null,"trade_payables":null,"short_term_debt":null,"long_term_debt":null,"total_debt":null,"shareholders_equity":null,"retained_earnings":null,"operating_cash_flow":null,"capex":null,"promoter_holding_pct":null}}
Text: {text}'''

def extract_financials_with_llm(text: str, **kwargs) -> dict[str, Any]:
    client = get_llm_client()
    if not client.is_available():
        logger.error("Ollama not running. Start with: ollama serve")
        return {}

    # Limit text so input + output stays within num_ctx 4096
    snippet = text[:3000]
    result: dict[str, Any] = {}

    try:
        part_a = client.chat_json(EXTRACTION_PROMPT_A.format(text=snippet), system=EXTRACTION_SYSTEM, temperature=0.0)
        if isinstance(part_a, dict):
            result.update(part_a)
    except Exception as e:
        logger.error(f"Extraction part A failed: {e}")

    try:
        part_b = client.chat_json(EXTRACTION_PROMPT_B.format(text=snippet), system=EXTRACTION_SYSTEM, temperature=0.0)
        if isinstance(part_b, dict):
            result.update(part_b)
    except Exception as e:
        logger.error(f"Extraction part B failed: {e}")

    logger.info(f"Extracted {len(result)} fields from document")
    return result

# ── Schema Mapping ────────────────────────────────────────────────────────────

SYNONYMS = {
    "net sales": "revenue", "operating revenue": "revenue",
    "total revenue": "revenue", "revenue from operations": "revenue",
    "profit after tax": "net_profit", "net income": "net_profit",
    "profit for the year": "net_profit", "total equity": "shareholders_equity",
    "net worth": "shareholders_equity",
}

UNIT_MULTIPLIERS = {"crores": 1.0, "lakhs": 0.01, "millions": 0.1, "thousands": 0.0001, "billions": 100.0}

def apply_schema_mapping(raw: dict[str, Any], custom: dict[str, str] | None = None) -> dict[str, Any]:
    mappings = {**SYNONYMS, **(custom or {})}
    unit = str(raw.get("currency_unit", "crores")).lower()
    mult = UNIT_MULTIPLIERS.get(unit, 1.0)
    result: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = mappings.get(key.lower().strip(), key.lower().replace(" ", "_"))
        if isinstance(value, (int, float)) and canonical != "fiscal_year":
            result[canonical] = round(value * mult, 4)
        else:
            result[canonical] = value
    return result