"""
agents/financial_agent.py

Computes financial ratios + extended ML feature set from ExtractedData.
Handles multi-year data for growth calculations.
"""
from __future__ import annotations
from typing import Optional
from loguru import logger


def _to_float(val) -> Optional[float]:
    """Coerce any value (str/int/float/None) to float. Handles commas like 4,250.00"""
    if val is None:
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def safe_div(numerator, denominator) -> Optional[float]:
    n, d = _to_float(numerator), _to_float(denominator)
    if n is None or d is None or d == 0:
        return None
    return round(n / d, 4)


def compute_ratios(current: dict, prior: dict | None = None) -> dict:
    ratios: dict = {
        "fiscal_year": current.get("fiscal_year"),
        "entity_id":   current.get("entity_id"),
    }

    # ── Raw floats used repeatedly ────────────────────────────────────────────
    revenue       = _to_float(current.get("revenue")) or _to_float(current.get("total_income"))
    net_profit    = _to_float(current.get("net_profit"))
    ebitda        = _to_float(current.get("ebitda"))
    ebit          = _to_float(current.get("ebit"))
    interest_exp  = _to_float(current.get("interest_expense"))
    depreciation  = _to_float(current.get("depreciation"))
    pbt           = _to_float(current.get("pbt"))
    tax           = _to_float(current.get("tax"))

    # Derive ebit/ebitda if missing — common when LLM extraction is truncated
    # ebit   = PBT + Interest
    # ebitda = EBIT + Depreciation
    if ebit is None and pbt is not None and interest_exp is not None:
        ebit = round(pbt + interest_exp, 4)
    if ebitda is None and ebit is not None and depreciation is not None:
        ebitda = round(ebit + depreciation, 4)
    total_assets  = _to_float(current.get("total_assets"))
    total_debt    = _to_float(current.get("total_debt"))
    equity        = _to_float(current.get("shareholders_equity"))
    curr_assets   = _to_float(current.get("current_assets"))
    curr_liab     = _to_float(current.get("current_liabilities"))
    inventories   = _to_float(current.get("inventories"))
    receivables   = _to_float(current.get("trade_receivables"))
    payables      = _to_float(current.get("trade_payables"))
    cash          = _to_float(current.get("cash_and_equivalents"))
    ocf           = _to_float(current.get("operating_cash_flow"))
    capex         = _to_float(current.get("capex"))
    lt_debt       = _to_float(current.get("long_term_debt"))
    st_debt       = _to_float(current.get("short_term_debt"))
    cogs          = _to_float(current.get("cost_of_goods_sold"))
    retained_earn = _to_float(current.get("retained_earnings"))
    tax           = _to_float(current.get("tax"))
    pbt           = _to_float(current.get("pbt"))
    promoter_pct  = _to_float(current.get("promoter_holding_pct"))

    # ── 1. Liquidity ──────────────────────────────────────────────────────────
    ratios["current_ratio"]     = safe_div(curr_assets, curr_liab)
    quick_assets                = _subtract(curr_assets, inventories)
    ratios["quick_ratio"]       = safe_div(quick_assets, curr_liab)
    ratios["cash_ratio"]        = safe_div(cash, curr_liab)
    # Net working capital as % of revenue
    nwc = _subtract(curr_assets, curr_liab)
    ratios["nwc_to_revenue"]    = _pct(safe_div(nwc, revenue))

    # ── 2. Leverage / Solvency ────────────────────────────────────────────────
    ratios["debt_to_equity"]       = safe_div(total_debt, equity)
    ratios["debt_to_assets"]       = safe_div(total_debt, total_assets)
    ratios["interest_coverage"]    = safe_div(ebit, interest_exp)
    st_debt_f   = st_debt or 0.0
    int_exp_f   = interest_exp or 0.0
    debt_service = int_exp_f + st_debt_f
    ratios["debt_service_coverage"] = safe_div(ebitda, debt_service or None)
    # Financial leverage = Total Assets / Equity
    ratios["financial_leverage"]   = safe_div(total_assets, equity)
    # Long-term debt ratio
    ratios["lt_debt_ratio"]        = safe_div(lt_debt, total_assets)
    # Equity multiplier
    ratios["equity_multiplier"]    = safe_div(total_assets, equity)
    # Net debt = Total Debt - Cash
    net_debt = _subtract(total_debt, cash)
    ratios["net_debt_to_ebitda"]   = safe_div(net_debt, ebitda)
    ratios["net_debt_to_equity"]   = safe_div(net_debt, equity)

    # ── 3. Profitability ──────────────────────────────────────────────────────
    gross_profit                   = _to_float(current.get("gross_profit"))
    ratios["gross_margin_pct"]     = _pct(safe_div(gross_profit, revenue))
    ratios["ebitda_margin_pct"]    = _pct(safe_div(ebitda, revenue))
    ratios["ebit_margin_pct"]      = _pct(safe_div(ebit, revenue))
    ratios["net_margin_pct"]       = _pct(safe_div(net_profit, revenue))
    ratios["roe"]                  = _pct(safe_div(net_profit, equity))
    ratios["roa"]                  = _pct(safe_div(net_profit, total_assets))
    cap_employed                   = _subtract(total_assets, curr_liab)
    ratios["roce"]                 = _pct(safe_div(ebit, cap_employed))
    # Tax rate
    ratios["effective_tax_rate"]   = _pct(safe_div(tax, pbt)) if pbt and pbt > 0 else None
    # Operating leverage proxy = EBIT / Revenue
    ratios["operating_leverage"]   = safe_div(ebit, revenue)

    # ── 4. Efficiency ─────────────────────────────────────────────────────────
    ratios["asset_turnover"]       = safe_div(revenue, total_assets)
    ratios["equity_turnover"]      = safe_div(revenue, equity)
    ratios["receivables_days"]     = _days(receivables, revenue)
    ratios["payables_days"]        = _days(payables, cogs or revenue)
    ratios["inventory_days"]       = _days(inventories, cogs or revenue)
    r_days = ratios["receivables_days"] or 0
    p_days = ratios["payables_days"] or 0
    i_days = ratios["inventory_days"] or 0
    ratios["cash_conversion_cycle"] = round(r_days + i_days - p_days, 2) if (r_days or i_days or p_days) else None
    # Fixed asset turnover
    fixed_assets = _subtract(total_assets, curr_assets)
    ratios["fixed_asset_turnover"] = safe_div(revenue, fixed_assets)
    # Working capital turnover
    ratios["wc_turnover"]          = safe_div(revenue, nwc) if nwc and nwc != 0 else None

    # ── 5. Cash Flow Quality ──────────────────────────────────────────────────
    ratios["ocf_to_revenue"]       = safe_div(ocf, revenue)
    ratios["ocf_to_net_profit"]    = safe_div(ocf, net_profit)          # accruals quality
    ratios["ocf_to_total_debt"]    = safe_div(ocf, total_debt)
    ratios["ocf_to_interest"]      = safe_div(ocf, interest_exp)
    ratios["free_cash_flow"]       = round((ocf or 0) - (capex or 0), 4) if ocf is not None else None
    ratios["capex_to_revenue"]     = _pct(safe_div(capex, revenue))
    ratios["capex_intensity"]      = safe_div(capex, fixed_assets) if fixed_assets else None
    # Cash earnings quality = OCF / EBITDA (>0.8 is good)
    ratios["cash_earnings_quality"]= safe_div(ocf, ebitda)

    # ── 6. Altman Z-Score (manufacturing variant) ─────────────────────────────
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    x1 = safe_div(nwc, total_assets)
    x2 = safe_div(retained_earn, total_assets)
    x3 = safe_div(ebit, total_assets)
    x4 = safe_div(equity, total_debt)
    x5 = safe_div(revenue, total_assets)
    if all(v is not None for v in [x1, x2, x3, x4, x5]):
        z = round(1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5, 4)
        ratios["altman_z_score"] = z
        ratios["altman_zone"] = "safe" if z > 2.99 else ("grey" if z > 1.81 else "distress")
    else:
        ratios["altman_z_score"] = None
        ratios["altman_zone"]    = None

    # ── 7. Shareholding ───────────────────────────────────────────────────────
    ratios["promoter_holding_pct"] = promoter_pct

    # ── 8. Growth (YoY) ───────────────────────────────────────────────────────
    if prior:
        prior_rev    = _to_float(prior.get("revenue")) or _to_float(prior.get("total_income"))
        prior_profit = _to_float(prior.get("net_profit"))
        prior_ebitda = _to_float(prior.get("ebitda"))
        prior_assets = _to_float(prior.get("total_assets"))
        ratios["revenue_growth_pct"]  = _growth_pct(revenue, prior_rev)
        ratios["profit_growth_pct"]   = _growth_pct(net_profit, prior_profit)
        ratios["ebitda_growth_pct"]   = _growth_pct(ebitda, prior_ebitda)
        ratios["asset_growth_pct"]    = _growth_pct(total_assets, prior_assets)
        # Debt trend: positive = debt increasing (risk signal)
        prior_debt = _to_float(prior.get("total_debt"))
        ratios["debt_growth_pct"]     = _growth_pct(total_debt, prior_debt)
    else:
        for k in ["revenue_growth_pct", "profit_growth_pct", "ebitda_growth_pct",
                  "asset_growth_pct", "debt_growth_pct"]:
            ratios[k] = None

    return ratios


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _subtract(a, b) -> Optional[float]:
    af = _to_float(a)
    if af is None:
        return None
    return af - (_to_float(b) or 0)

def _pct(ratio: Optional[float]) -> Optional[float]:
    if ratio is None:
        return None
    return round(ratio * 100, 4)

def _days(balance, flow) -> Optional[float]:
    r = safe_div(balance, flow)
    return round(r * 365, 2) if r is not None else None

def _growth_pct(curr_val, prior_val) -> Optional[float]:
    c, p = _to_float(curr_val), _to_float(prior_val)
    if c is None or p is None or p == 0:
        return None
    return round(((c - p) / abs(p)) * 100, 4)


# ─── Analysis Narrative ───────────────────────────────────────────────────────

def interpret_ratios(ratios: dict) -> list[dict]:
    flags = []

    def flag(metric, value, condition_good, condition_warn, fmt=None):
        if value is None:
            return
        status  = "good" if condition_good(value) else ("warning" if condition_warn(value) else "critical")
        display = f"{value:.2f}%" if fmt == "pct" else f"{value:.2f}"
        flags.append({"metric": metric, "value": value, "display": display, "status": status,
                      "comment": _ratio_comment(metric, value, status)})

    flag("Current Ratio",       ratios.get("current_ratio"),        lambda v: v >= 1.5, lambda v: v >= 1.0)
    flag("Quick Ratio",         ratios.get("quick_ratio"),           lambda v: v >= 1.0, lambda v: v >= 0.7)
    flag("Cash Ratio",          ratios.get("cash_ratio"),            lambda v: v >= 0.5, lambda v: v >= 0.2)
    flag("Debt-to-Equity",      ratios.get("debt_to_equity"),        lambda v: v <= 1.5, lambda v: v <= 3.0)
    flag("Interest Coverage",   ratios.get("interest_coverage"),     lambda v: v >= 3.0, lambda v: v >= 1.5)
    flag("DSCR",                ratios.get("debt_service_coverage"), lambda v: v >= 1.5, lambda v: v >= 1.0)
    flag("Net Margin %",        ratios.get("net_margin_pct"),        lambda v: v >= 10, lambda v: v >= 0,   fmt="pct")
    flag("EBITDA Margin %",     ratios.get("ebitda_margin_pct"),     lambda v: v >= 15, lambda v: v >= 5,   fmt="pct")
    flag("ROE %",               ratios.get("roe"),                   lambda v: v >= 15, lambda v: v >= 5,   fmt="pct")
    flag("ROA %",               ratios.get("roa"),                   lambda v: v >= 8,  lambda v: v >= 2,   fmt="pct")
    flag("ROCE %",              ratios.get("roce"),                  lambda v: v >= 15, lambda v: v >= 8,   fmt="pct")
    flag("Revenue Growth %",    ratios.get("revenue_growth_pct"),    lambda v: v >= 10, lambda v: v >= 0,   fmt="pct")
    flag("Altman Z-Score",      ratios.get("altman_z_score"),        lambda v: v > 2.99, lambda v: v > 1.81)
    flag("Net Debt/EBITDA",     ratios.get("net_debt_to_ebitda"),    lambda v: v <= 2.0, lambda v: v <= 4.0)
    flag("OCF/Net Profit",      ratios.get("ocf_to_net_profit"),     lambda v: v >= 1.0, lambda v: v >= 0.7)
    flag("Cash Earnings Quality", ratios.get("cash_earnings_quality"), lambda v: v >= 0.8, lambda v: v >= 0.5)
    return flags


def _ratio_comment(metric: str, value: float, status: str) -> str:
    comments = {
        ("Current Ratio",        "good"):     "Adequate short-term liquidity.",
        ("Current Ratio",        "warning"):  "Liquidity is tight; monitor working capital.",
        ("Current Ratio",        "critical"): "Insufficient current assets to cover current liabilities.",
        ("Debt-to-Equity",       "good"):     "Comfortable leverage level.",
        ("Debt-to-Equity",       "warning"):  "Elevated leverage; warrants monitoring.",
        ("Debt-to-Equity",       "critical"): "Highly leveraged; significant repayment risk.",
        ("Interest Coverage",    "good"):     "Strong ability to service interest obligations.",
        ("Interest Coverage",    "warning"):  "Thin interest coverage margin.",
        ("Interest Coverage",    "critical"): "Unable to comfortably cover interest from operating profit.",
        ("Net Margin %",         "good"):     "Healthy profitability.",
        ("Net Margin %",         "warning"):  "Slim margins; vulnerability to cost pressure.",
        ("Net Margin %",         "critical"): "Loss-making or near-zero profitability.",
        ("Altman Z-Score",       "good"):     "Company in safe zone — low bankruptcy risk.",
        ("Altman Z-Score",       "warning"):  "Company in grey zone — elevated distress risk.",
        ("Altman Z-Score",       "critical"): "Company in distress zone — high default probability.",
        ("Net Debt/EBITDA",      "good"):     "Debt well-covered by operating earnings.",
        ("Net Debt/EBITDA",      "warning"):  "Leverage is elevated relative to earnings.",
        ("Net Debt/EBITDA",      "critical"): "Debt significantly exceeds operational earnings capacity.",
        ("OCF/Net Profit",       "good"):     "Earnings are backed by strong cash conversion.",
        ("OCF/Net Profit",       "warning"):  "Some gap between reported profits and cash generation.",
        ("OCF/Net Profit",       "critical"): "Earnings quality is poor — profits not converting to cash.",
        ("Cash Earnings Quality","good"):     "High cash conversion from EBITDA.",
        ("Cash Earnings Quality","warning"):  "Moderate cash conversion; monitor working capital changes.",
        ("Cash Earnings Quality","critical"): "Low cash conversion; potential accruals or WC absorption.",
    }
    return comments.get((metric, status), f"{metric} is {status}.")