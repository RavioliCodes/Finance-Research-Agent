"""
query_expander.py
Expands user queries into multiple variations for better retrieval coverage.
Uses synonym expansion, acronym resolution, temporal expansion, and
question decomposition — tuned for financial / 10-K documents.
"""

import re


class QueryExpander:
    """
    Generates query variations to improve retrieval recall.
    Techniques:
    - Synonym/keyword expansion
    - Business acronym resolution
    - Temporal phrase expansion (quarters, fiscal years)
    - Question decomposition
    - Metric-specific query generation
    """

    # Business domain synonyms
    SYNONYMS = {
        "revenue": ["revenue", "sales", "income", "earnings", "top line", "net revenue", "total revenue"],
        "profit": ["profit", "net income", "margin", "earnings", "bottom line", "ebitda", "operating income"],
        "growth": ["growth", "increase", "trend", "trajectory", "year-over-year", "yoy", "change"],
        "risk": ["risk", "challenge", "threat", "concern", "issue", "vulnerability"],
        "underperforming": ["underperforming", "below target", "declining", "lagging",
                            "behind", "miss", "gap", "weakness"],
        "department": ["department", "team", "division", "segment", "unit", "function"],
        "strategy": ["strategy", "plan", "initiative", "roadmap", "pillar", "goal"],
        "customer": ["customer", "client", "account", "user", "enterprise"],
        "employee": ["employee", "workforce", "headcount", "staff", "talent", "hiring"],
        "cost": ["cost", "expense", "spend", "expenditure", "budget", "opex", "cost of revenue"],
        "asset": ["asset", "holding", "resource", "property", "total assets"],
        "liability": ["liability", "debt", "obligation", "payable", "total liabilities"],
        "equity": ["equity", "ownership", "net worth", "stockholders equity", "shareholders equity"],
        "investment": ["investment", "allocation", "funding", "capital deployment", "stake"],
        "subscription": ["subscription", "recurring revenue", "ARR", "annualized recurring revenue"],
        "cash flow": ["cash flow", "cash from operations", "operating cash flow", "free cash flow"],
        "operating expense": ["operating expense", "opex", "research and development", "sales and marketing"],
    }

    ACRONYMS = {
        "arr": "annual recurring revenue",
        "mrr": "monthly recurring revenue",
        "nrr": "net revenue retention",
        "nps": "net promoter score",
        "cac": "customer acquisition cost",
        "ltv": "lifetime value",
        "saas": "software as a service",
        "yoy": "year over year",
        "qoq": "quarter over quarter",
        "ebitda": "earnings before interest taxes depreciation amortization",
        "opex": "operating expenses",
        "capex": "capital expenditure",
        "sla": "service level agreement",
        "kpi": "key performance indicator",
        "roi": "return on investment",
        "pat": "profit after tax",
        "eps": "earnings per share",
        "gaap": "generally accepted accounting principles",
        "fy": "fiscal year",
        "cogs": "cost of goods sold",
    }

    # Temporal phrase mappings — maps natural language to 10-K phrasing.
    _QUARTER_NAMES = {
        "first": "1", "second": "2", "third": "3", "fourth": "4",
        "1st": "1", "2nd": "2", "3rd": "3", "4th": "4",
        "last": "4",
    }

    def expand(self, question: str) -> list[str]:
        """
        Generate multiple query variations from a single question.
        Returns the original + expanded versions.
        """
        variations = [question]

        resolved = self._resolve_acronyms(question)
        if resolved != question:
            variations.append(resolved)

        temporal_queries = self._expand_temporal(question)
        variations.extend(temporal_queries)

        keyword_queries = self._expand_keywords(question)
        variations.extend(keyword_queries)

        sub_questions = self._decompose_question(question)
        variations.extend(sub_questions)

        seen = set()
        unique = []
        for v in variations:
            v_lower = v.lower().strip()
            if v_lower not in seen:
                seen.add(v_lower)
                unique.append(v)

        return unique[:6] #max 6 variations to avoid overwhelming the retriever

    def _resolve_acronyms(self, text: str) -> str:
        """Replace known acronyms with full forms."""
        words = text.split()
        resolved = []
        for w in words:
            clean = re.sub(r'[^a-zA-Z]', '', w).lower()
            if clean in self.ACRONYMS:
                resolved.append(self.ACRONYMS[clean])
            else:
                resolved.append(w)
        return " ".join(resolved)

    def _expand_temporal(self, question: str) -> list[str]:
        """Expand temporal references into financial report phrasing.

        e.g. "last quarter of 2025"  →  "Q4 2025", "fourth quarter 2025",
             "three months ended 2025", "fiscal 2025"
        """
        q_lower = question.lower()
        expansions: list[str] = []

        # Match patterns like "last quarter of 2025", "Q3 2024", "first quarter 2025"
        quarter_re = re.compile(
            r"(?:the\s+)?"
            r"(?:(first|second|third|fourth|1st|2nd|3rd|4th|last)\s+quarter"
            r"(?:\s+of)?\s+(\d{4}))"
            r"|(?:q([1-4])\s*(\d{4}))",
            re.IGNORECASE,
        )

        m = quarter_re.search(q_lower)
        if m:
            if m.group(1):  # "first quarter of 2025" style
                qnum = self._QUARTER_NAMES.get(m.group(1).lower(), m.group(1))
                year = m.group(2)
            else:  # "Q3 2024" style
                qnum = m.group(3)
                year = m.group(4)

            original_span = m.group(0)
            alt_phrases = [
                f"Q{qnum} {year}",
                f"Q{qnum} fiscal {year}",
                f"fourth quarter {year}" if qnum == "4" else f"quarter {qnum} {year}",
                f"three months ended {year}",
                f"fiscal {year}",
            ]
            for phrase in alt_phrases:
                rewritten = q_lower.replace(original_span, phrase)
                if rewritten != q_lower:
                    expansions.append(rewritten)

        # Handle "fiscal year 2025" / "FY2025" / "FY 2025"
        fy_re = re.compile(r"(?:fiscal\s+year|fy)\s*(\d{4})", re.IGNORECASE)
        m_fy = fy_re.search(q_lower)
        if m_fy:
            year = m_fy.group(1)
            for alt in [f"fiscal {year}", f"FY{year}", f"year ended {year}"]:
                rewritten = fy_re.sub(alt, q_lower)
                if rewritten != q_lower:
                    expansions.append(rewritten)

        # Generic: if "2025" appears but no quarter match, add fiscal variants
        if not m and not m_fy and re.search(r"\b20\d{2}\b", q_lower):
            year_match = re.search(r"\b(20\d{2})\b", q_lower)
            if year_match:
                year = year_match.group(1)
                expansions.append(q_lower + f" fiscal {year}")

        return expansions[:3]

    def _expand_keywords(self, question: str) -> list[str]:
        """Generate variations using synonym expansion."""
        q_lower = question.lower()
        expansions = []

        for key, synonyms in self.SYNONYMS.items():
            if key in q_lower:
                for syn in synonyms:
                    if syn != key and syn not in q_lower:
                        expanded = q_lower.replace(key, syn)
                        expansions.append(expanded)
                        if len(expansions) >= 2:
                            return expansions

        return expansions

    def _decompose_question(self, question: str) -> list[str]:
        """Break compound questions into sub-questions."""
        q_lower = question.lower()
        sub_qs = []

        if " and " in q_lower:
            parts = q_lower.split(" and ")
            if len(parts) == 2 and len(parts[0]) > 15:
                sub_qs.extend(parts)

        return sub_qs[:2]
