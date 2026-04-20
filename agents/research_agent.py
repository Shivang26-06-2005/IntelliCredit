"""
agents/research_agent.py

Fetches external intelligence about a company:
  - News articles via NewsAPI (free tier: 100 req/day)
  - Sentiment scoring via local Ollama model (free, no API key needed)

Returns a list of ResearchFinding-compatible dicts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings
from utils.llm_client import get_llm_client

settings = get_settings()

SENTIMENT_SYSTEM = """You are a financial news sentiment analyzer.
Return ONLY a JSON object. No explanation. No markdown."""

SENTIMENT_PROMPT = """Analyze this news about a company for credit risk assessment.

Return ONLY this JSON:
{{
  "sentiment": "positive",
  "score": 0.0,
  "key_topics": [],
  "risk_relevant": false
}}

Rules:
- sentiment: "positive", "neutral", or "negative"
- score: float from -1.0 (very negative) to 1.0 (very positive)
- key_topics: up to 3 from: debt, litigation, revenue, fraud, governance, management, regulatory, market
- risk_relevant: true if news could affect ability to repay loans

Headline: {headline}
Summary: {summary}"""


class ResearchAgent:
    def __init__(self, news_api_key: str | None = None):
        self.news_api_key = news_api_key or settings.news_api_key
        self.ollama = get_llm_client()

    # ─── News Fetch ───────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_news(self, company_name: str, max_articles: int = 20) -> list[dict[str, Any]]:
        """Fetch recent news articles via NewsAPI."""
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not set. Skipping news fetch. Add it to .env for research.")
            return []

        params = {
            "q": f'"{company_name}"',
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": max_articles,
            "apiKey": self.news_api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://newsapi.org/v2/everything", params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for art in data.get("articles", []):
            articles.append({
                "headline": art.get("title", ""),
                "summary": art.get("description", "") or art.get("content", "")[:300],
                "source_url": art.get("url", ""),
                "source_type": "news",
                "published_at": _parse_date(art.get("publishedAt")),
            })
        return articles

    # ─── Sentiment Analysis via Ollama ────────────────────────────────────────

    def analyze_sentiment(self, headline: str, summary: str) -> dict[str, Any]:
        """Score sentiment of a single article using local Ollama model."""
        default = {"sentiment": "neutral", "score": 0.0, "key_topics": [], "risk_relevant": False}

        if not self.ollama.is_available():
            logger.warning("Ollama not available for sentiment. Using neutral default.")
            return default

        prompt = SENTIMENT_PROMPT.format(
            headline=headline[:300],
            summary=(summary or "")[:400],
        )
        try:
            result = self.ollama.chat_json(prompt, system=SENTIMENT_SYSTEM, temperature=0.0)
            # Validate expected keys are present
            if "sentiment" in result and "score" in result:
                return result
            return default
        except Exception as e:
            logger.error(f"Ollama sentiment analysis failed: {e}")
            return default

    # ─── Main Research Run ────────────────────────────────────────────────────

    async def run(self, company_name: str, sector: str | None = None) -> list[dict[str, Any]]:
        """
        Full research pipeline: fetch news → score sentiment → return findings.
        Sentiment scoring runs synchronously (Ollama is a local HTTP call).
        """
        logger.info(f"Research agent starting for: {company_name}")
        articles = await self.fetch_news(company_name)

        findings = []
        for art in articles:
            sentiment_data = self.analyze_sentiment(
                art.get("headline", ""),
                art.get("summary", ""),
            )
            findings.append({
                **art,
                "sentiment": sentiment_data.get("sentiment", "neutral"),
                "sentiment_score": sentiment_data.get("score", 0.0),
                "relevance_score": 1.0 if sentiment_data.get("risk_relevant") else 0.5,
                "key_topics": sentiment_data.get("key_topics", []),
            })

        logger.info(f"Research agent completed: {len(findings)} findings for {company_name}")
        return findings

    # ─── Risk Signal Extraction ───────────────────────────────────────────────

    def extract_risk_signals(self, findings: list[dict]) -> list[dict]:
        """Filter negative findings into categorized risk signals."""
        RISK_KEYWORDS = {
            "litigation":  ["court", "lawsuit", "legal", "sue", "penalty", "sebi", "ed", "cbi"],
            "debt_stress": ["default", "npa", "restructur", "debt trap", "overdue"],
            "governance":  ["fraud", "scam", "irregularit", "audit qualif", "restat"],
            "management":  ["ceo resign", "md resign", "board resign", "key person"],
            "market":      ["market share loss", "competition", "revenue decline", "order cancel"],
            "regulatory":  ["rbi action", "sebi action", "show cause", "licence cancel"],
        }

        signals = []
        for finding in findings:
            if finding.get("sentiment") != "negative":
                continue
            text = (
                (finding.get("headline") or "") + " " + (finding.get("summary") or "")
            ).lower()
            for category, keywords in RISK_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    signals.append({
                        "category": category,
                        "headline": finding.get("headline"),
                        "source_url": finding.get("source_url"),
                        "published_at": str(finding.get("published_at", "")),
                        "sentiment_score": finding.get("sentiment_score", 0),
                    })
                    break
        return signals


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None