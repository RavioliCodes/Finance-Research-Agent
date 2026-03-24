"""
answer_generator.py
Generates concise, factual answers grounded in retrieved document chunks.

Supports three modes:
1. LOCAL mode (default): Extractive + template-based, no API key needed.
2. API mode with OpenAI (default provider): Set LLM_PROVIDER=openai
3. API mode with Anthropic: Set LLM_PROVIDER=anthropic

API keys and config are read from environment variables (or a .env file).
See .env.example for all available settings.
"""

import os
import re
from dataclasses import dataclass
from vector_store import SearchResult

# # Load .env file automatically if present (requires python-dotenv).
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass


@dataclass
class AgentResponse:
    """The agent's final response to a leadership question."""
    answer: str
    confidence: str           # "high", "medium", "low"
    sources: list[dict]       # source references
    context_used: str         # concatenated context for transparency
    query_variations: list[str]


class AnswerGenerator:
    """
    Generates answers from retrieved context.
    """

    def __init__(self, mode: str = "local"):
        """
        Args:
            mode: 'local' for template-based, 'api' for LLM API.
                  Provider is selected via the LLM_PROVIDER env var
                  ('openai' by default, or 'anthropic').
        """
        self.mode = mode
        self.provider = os.environ.get("LLM_PROVIDER", "openai").lower()

        # OpenAI
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_api_url = os.environ.get(
            "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
        )

        # Anthropic
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.anthropic_api_url = os.environ.get(
            "ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages"
        )

    def generate(
        self,
        question: str,
        results: list[SearchResult],
        query_variations: list[str],
    ) -> AgentResponse:
        """Generate an answer from search results."""
        if not results:
            return AgentResponse(
                answer="I could not find relevant information in the company documents to answer this question. Please ensure the relevant documents have been uploaded.",
                confidence="low",
                sources=[],
                context_used="",
                query_variations=query_variations,
            )

        # Build context
        context = self._build_context(results)
        sources = self._extract_sources(results)
        confidence = self._assess_confidence(results)

        if self.mode == "api":
            answer = self._generate_via_llm(question, context, results)
        else:
            answer = self._generate_local(question, results, context)

        return AgentResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            context_used=context,
            query_variations=query_variations,
        )


    def _generate_local(
        self, question: str, results: list[SearchResult], context: str
    ) -> str:
        """
        Extractive answer generation using keyword matching and
        intelligent summarization heuristics.
        """
        question_lower = question.lower()

        q_type = self._classify_question(question_lower)

        key_sentences = self._extract_key_sentences(question_lower, results)

        # Build structured answer
        if q_type == "metric":
            answer = self._format_metric_answer(question_lower, key_sentences, results)
        elif q_type == "risk":
            answer = self._format_risk_answer(key_sentences, results)
        elif q_type == "performance":
            answer = self._format_performance_answer(key_sentences, results)
        elif q_type == "strategy":
            answer = self._format_strategy_answer(key_sentences, results)
        else:
            answer = self._format_general_answer(question, key_sentences, results)

        return answer

    def _classify_question(self, q: str) -> str:
        """Classify the question type for tailored response formatting."""
        metric_kw = ["revenue", "profit", "margin", "cost", "cash", "arr", "growth", "ebitda", "financial"]
        risk_kw = ["risk", "challenge", "threat", "concern", "issue", "problem", "danger"]
        perf_kw = ["performance", "underperform", "performing", "department", "team", "kpi", "metric", "status"]
        strat_kw = ["strategy", "plan", "future", "goal", "target", "initiative", "roadmap", "pillar"]

        if any(kw in q for kw in metric_kw):
            return "metric"
        if any(kw in q for kw in risk_kw):
            return "risk"
        if any(kw in q for kw in perf_kw):
            return "performance"
        if any(kw in q for kw in strat_kw):
            return "strategy"
        return "general"

    def _extract_key_sentences(
        self, question: str, results: list[SearchResult]
    ) -> list[str]:
        """Extract sentences most relevant to the question."""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        # remove stop words
        stop = {"what", "is", "our", "the", "a", "an", "in", "of", "to", "and",
                "are", "were", "was", "be", "has", "have", "had", "do", "does",
                "did", "will", "would", "could", "should", "which", "that",
                "this", "for", "from", "with", "about", "how", "when", "where",
                "who", "why", "current", "last", "latest", "recent"}
        question_words -= stop

        scored_sentences = []
        for r in results:
            sentences = re.split(r'(?<=[.!?])\s+', r.chunk.text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                sent_words = set(re.findall(r'\b\w+\b', sent.lower()))
                overlap = len(question_words & sent_words)
                # boost sentences with numbers (metrics/facts)
                has_numbers = bool(re.search(r'\$?\d+[\d,.]*%?', sent))
                score = overlap + (1.5 if has_numbers else 0) + (r.score * 2)
                scored_sentences.append((score, sent, r.chunk.source_file))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored_sentences[:15]]

    def _format_metric_answer(self, question, sentences, results):
        """Format answer for metric/financial questions."""
        lines = ["Based on the company documents:\n"]
        # Extract sentences containing numbers
        metric_sentences = [s for s in sentences if re.search(r'\$?\d+[\d,.]*%?', s)]
        other_sentences = [s for s in sentences if s not in metric_sentences]

        if metric_sentences:
            lines.append("**Key Figures:**")
            for s in metric_sentences[:6]:
                lines.append(f"• {s}")

        if other_sentences[:3]:
            lines.append("\n**Context:**")
            for s in other_sentences[:3]:
                lines.append(f"• {s}")

        return "\n".join(lines)

    def _format_risk_answer(self, sentences, results):
        """Format answer for risk-related questions."""
        lines = ["Based on the company documents, the following risks and challenges have been identified:\n"]
        risk_sentences = [s for s in sentences if any(
            kw in s.lower() for kw in ["risk", "challenge", "threat", "concern", "delay",
                                        "drop", "decrease", "below", "miss", "gap",
                                        "competition", "competitor", "uncertainty"]
        )]
        if risk_sentences:
            for i, s in enumerate(risk_sentences[:8], 1):
                lines.append(f"{i}. {s}")
        else:
            for i, s in enumerate(sentences[:6], 1):
                lines.append(f"{i}. {s}")
        return "\n".join(lines)

    def _format_performance_answer(self, sentences, results):
        """Format answer for performance/department questions."""
        lines = ["Based on the company documents:\n"]
        for s in sentences[:8]:
            lines.append(f"• {s}")
        return "\n".join(lines)

    def _format_strategy_answer(self, sentences, results):
        """Format answer for strategy questions."""
        lines = ["Based on the strategic documents:\n"]
        for s in sentences[:8]:
            lines.append(f"• {s}")
        return "\n".join(lines)

    def _format_general_answer(self, question, sentences, results):
        """Format answer for general questions."""
        lines = [f"Based on the company documents:\n"]
        for s in sentences[:8]:
            lines.append(f"• {s}")
        return "\n".join(lines)


    #  API-based generation

    _SYSTEM_PROMPT = (
        "You are a Financial Analysis Agent that answers leadership questions "
        "based ONLY on the provided document context (typically SEC filings, "
        "10-K/10-Q reports, earnings reports, and internal financial documents).\n\n"
        "Rules:\n"
        "1. ONLY use information explicitly stated in the provided context. "
        "Do NOT use general knowledge or make assumptions.\n"
        "2. When the context contains relevant numbers, cite them precisely "
        "(dollar amounts, percentages, growth rates, period comparisons).\n"
        "3. Pay close attention to fiscal periods — companies may have non-calendar "
        "fiscal years (e.g., fiscal year ending in November or March).\n"
        "4. When the context contains tables, read them carefully — column headers "
        "indicate what each value represents.\n"
        "5. If the context does NOT contain sufficient information to answer the "
        "question, say so explicitly and describe what information IS available.\n"
        "6. Structure your answer with clear sections and bullet points when the "
        "answer involves multiple data points.\n"
        "7. Always specify the time period and source section for cited data.\n"
        "8. Distinguish between GAAP and non-GAAP figures when both are present."
    )

    def _user_prompt(self, question: str, context: str) -> str:
        return (
            "Below are excerpts retrieved from the company's financial documents. "
            "Each excerpt is labeled with its source file, section, and relevance score.\n\n"
            "IMPORTANT: Read ALL excerpts carefully before answering. Financial data "
            "may be spread across multiple excerpts. Tables use Markdown formatting — "
            "pay attention to column headers.\n\n"
            f"--- BEGIN DOCUMENT CONTEXT ---\n{context}\n--- END DOCUMENT CONTEXT ---\n\n"
            f"Question: {question}\n\n"
            "Provide a thorough, data-driven answer grounded exclusively in the "
            "document context above. Cite specific figures and their time periods."
        )

    def _generate_via_llm(
        self, question: str, context: str, results: list[SearchResult]
    ) -> str:
        """Dispatch to the configured LLM provider, fall back to local on failure."""
        if self.provider == "anthropic" and self.anthropic_api_key:
            return self._generate_via_anthropic(question, context, results)
        if self.openai_api_key:
            return self._generate_via_openai(question, context, results)
        # No API key available — degrade gracefully.
        return self._generate_local(question, results, context)

    def _generate_via_openai(
        self, question: str, context: str, results: list[SearchResult]
    ) -> str:
        try:
            import requests
        except ImportError:
            return self._generate_local(question, results, context)

        try:
            resp = requests.post(
                self.openai_api_url,
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.openai_model,
                    "messages": [
                        {"role": "system", "content": self._SYSTEM_PROMPT},
                        {"role": "user", "content": self._user_prompt(question, context)},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return (
                f"OpenAI call failed ({e}). Falling back to local generation.\n\n"
                + self._generate_local(question, results, context)
            )

    def _generate_via_anthropic(
        self, question: str, context: str, results: list[SearchResult]
    ) -> str:
        try:
            import requests
        except ImportError:
            return self._generate_local(question, results, context)

        try:
            resp = requests.post(
                self.anthropic_api_url,
                headers={
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.anthropic_model,
                    "max_tokens": 4096,
                    "system": self._SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": self._user_prompt(question, context)},
                    ],
                },
                timeout=30,
            )
            if not resp.ok:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text}")
            return resp.json()["content"][0]["text"]
        except Exception as e:
            return (
                f"Anthropic call failed ({e}). Falling back to local generation.\n\n"
                + self._generate_local(question, results, context)
            )


    #  Helpers

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results with rich metadata."""
        parts = []
        for i, r in enumerate(results, 1):
            header_parts = [f"Source: {r.chunk.source_file}"]
            if r.chunk.section_header:
                header_parts.append(f"Section: {r.chunk.section_header}")
            page = r.chunk.metadata.get("page")
            if page:
                header_parts.append(f"Page: {page}")
            header_parts.append(f"Relevance: {r.score:.2f}")
            header = " | ".join(header_parts)
            parts.append(f"[Excerpt {i} — {header}]\n{r.chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, results: list[SearchResult]) -> list[dict]:
        """Extract source references."""
        sources = []
        seen = set()
        for r in results:
            key = (r.chunk.source_file, r.chunk.section_header)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": r.chunk.source_file,
                    "section": r.chunk.section_header or "General",
                    "relevance_score": round(r.score, 3),
                })
        return sources

    def _assess_confidence(self, results: list[SearchResult]) -> str:
        """Assess answer confidence using multiple signals.

        The scoring is scale-agnostic — it works with both raw cosine
        similarity scores (0–1) and RRF fusion scores (~0.01–0.1) by
        looking at relative score distribution rather than absolute
        thresholds.
        """
        if not results:
            return "low"

        scores = [r.score for r in results]
        top_score = scores[0]
        n_results = len(results)

        # -- Signal 1: result count (more hits = more evidence) --
        count_score = min(n_results / 5.0, 1.0)

        # -- Signal 2: top score relative to the score range --
        if n_results >= 2:
            score_range = top_score - scores[-1]
            dominance = (top_score - scores[-1]) / top_score if top_score > 0 else 0
        else:
            dominance = 0.5

        # -- Signal 3: score concentration —
        # High concentration means the retriever found focused matches.
        top3_sum = sum(scores[:3])
        total_sum = sum(scores) or 1e-9
        concentration = top3_sum / total_sum

        # -- Signal 4: multiple sources agreeing --
        unique_sources = len(set(r.chunk.source_file for r in results[:5]))
        source_score = min(unique_sources / 2.0, 1.0)

        # -- Combine signals --
        composite = (
            0.35 * count_score
            + 0.25 * dominance
            + 0.20 * concentration
            + 0.20 * source_score
        )

        if composite >= 0.55:
            return "high"
        elif composite >= 0.35:
            return "medium"
        return "low"
