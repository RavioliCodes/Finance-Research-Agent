"""
tools.py
Defines the tools available to the autonomous reasoning agent.

Each tool wraps functionality from the existing LeadershipInsightAgent
(vector store, query expander, etc.) and exposes it as a callable
action the ReAct loop can invoke.
"""

import re
import json
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Result returned by a tool execution."""
    tool_name: str
    success: bool
    output: str
    data: dict = field(default_factory=dict)


# ── Tool schemas (shared base for Anthropic / OpenAI native tool use) ─────────

_TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": (
            "Search the indexed financial documents for information relevant to a query. "
            "Use this to find specific facts, figures, metrics, or qualitative statements. "
            "Returns the most relevant excerpts with source and section metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — be specific (e.g., 'total revenue fiscal 2024' not just 'revenue')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_sections",
        "description": (
            "Search for information across two different topics or time periods and present "
            "them side by side for comparison. Useful for year-over-year analysis, segment "
            "comparisons, or contrasting metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_a": {
                    "type": "string",
                    "description": "First topic/period to search for (e.g., 'revenue fiscal 2024')",
                },
                "query_b": {
                    "type": "string",
                    "description": "Second topic/period to search for (e.g., 'revenue fiscal 2023')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Results per query",
                    "default": 5,
                },
            },
            "required": ["query_a", "query_b"],
        },
    },
    {
        "name": "extract_metrics",
        "description": (
            "Search for a specific financial metric and extract all numerical values found. "
            "Returns structured data with the numbers, their context, and source locations. "
            "Good for gathering precise figures before doing calculations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "The metric to extract (e.g., 'operating income', 'total assets', 'subscription revenue')",
                },
            },
            "required": ["metric"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Perform a numerical calculation. Supports basic arithmetic: +, -, *, /, "
            "as well as percentage change. Use this after extracting metrics to compute "
            "ratios, growth rates, or margins."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python arithmetic expression (e.g., '(19.06 - 17.61) / 17.61 * 100')",
                },
                "label": {
                    "type": "string",
                    "description": "A short label describing what this calculates (e.g., 'YoY revenue growth %')",
                },
            },
            "required": ["expression", "label"],
        },
    },
    {
        "name": "summarize_findings",
        "description": (
            "Review everything in working memory so far and produce a structured summary. "
            "Use this as an intermediate step when you've gathered enough raw data and need "
            "to organize your findings before forming a final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "final_answer",
        "description": (
            "Provide your final, comprehensive answer to the research question. "
            "Call this tool when you have gathered enough evidence to answer. "
            "Include specific numbers, time periods, and document sections in your answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your comprehensive, evidence-based answer to the question",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Your confidence level based on the evidence found",
                },
            },
            "required": ["answer", "confidence"],
        },
    },
]


def get_anthropic_tools() -> list[dict]:
    """Return tool definitions in Anthropic's native tool use format."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in _TOOL_SCHEMAS
    ]


def get_openai_tools() -> list[dict]:
    """Return tool definitions in OpenAI's function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in _TOOL_SCHEMAS
    ]


# ── Tool execution ────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes tools using the existing LeadershipInsightAgent's internals.
    """

    def __init__(self, agent):
        """
        Args:
            agent: An initialized LeadershipInsightAgent instance.
        """
        self.agent = agent
        self.vector_store = agent.vector_store
        self.query_expander = agent.query_expander

    def execute(self, tool_name: str, parameters: dict) -> ToolResult:
        """Dispatch to the appropriate tool handler."""
        handlers = {
            "search_documents": self._search_documents,
            "compare_sections": self._compare_sections,
            "extract_metrics": self._extract_metrics,
            "calculate": self._calculate,
            "summarize_findings": self._summarize_findings,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=f"Unknown tool: {tool_name}. Available tools: {list(handlers.keys())}",
            )
        try:
            return handler(parameters)
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=f"Tool '{tool_name}' failed: {e}",
            )

    # ── Individual tool handlers ──────────────────────────────────────────────

    def _search_documents(self, params: dict) -> ToolResult:
        query = params.get("query", "")
        top_k = int(params.get("top_k", 8))

        variations = self.query_expander.expand(query)
        if len(variations) > 1:
            results = self.vector_store.multi_query_search(variations, top_k=top_k)
        else:
            results = self.vector_store.search(query, top_k=top_k)

        results = self.vector_store.expand_context(results, window=1)

        if not results:
            return ToolResult(
                tool_name="search_documents",
                success=True,
                output="No relevant results found for this query.",
            )

        lines = []
        for i, r in enumerate(results, 1):
            section = r.chunk.section_header or "General"
            page = r.chunk.metadata.get("page", "?")
            lines.append(
                f"[Result {i} | {r.chunk.source_file} | Section: {section} | "
                f"Page: {page} | Score: {r.score:.3f}]\n{r.chunk.text}"
            )

        return ToolResult(
            tool_name="search_documents",
            success=True,
            output="\n\n---\n\n".join(lines),
            data={"num_results": len(results), "top_score": results[0].score},
        )

    def _compare_sections(self, params: dict) -> ToolResult:
        query_a = params.get("query_a", "")
        query_b = params.get("query_b", "")
        top_k = int(params.get("top_k", 5))

        # Search both
        vars_a = self.query_expander.expand(query_a)
        vars_b = self.query_expander.expand(query_b)

        results_a = (
            self.vector_store.multi_query_search(vars_a, top_k=top_k)
            if len(vars_a) > 1
            else self.vector_store.search(query_a, top_k=top_k)
        )
        results_b = (
            self.vector_store.multi_query_search(vars_b, top_k=top_k)
            if len(vars_b) > 1
            else self.vector_store.search(query_b, top_k=top_k)
        )

        lines = [f"=== Results for: {query_a} ===\n"]
        for i, r in enumerate(results_a, 1):
            section = r.chunk.section_header or "General"
            lines.append(f"[{i}] ({section}, score {r.score:.3f}) {r.chunk.text[:500]}")

        lines.append(f"\n=== Results for: {query_b} ===\n")
        for i, r in enumerate(results_b, 1):
            section = r.chunk.section_header or "General"
            lines.append(f"[{i}] ({section}, score {r.score:.3f}) {r.chunk.text[:500]}")

        return ToolResult(
            tool_name="compare_sections",
            success=True,
            output="\n".join(lines),
            data={
                "results_a_count": len(results_a),
                "results_b_count": len(results_b),
            },
        )

    def _extract_metrics(self, params: dict) -> ToolResult:
        metric = params.get("metric", "")

        variations = self.query_expander.expand(metric)
        results = self.vector_store.multi_query_search(variations, top_k=10)
        results = self.vector_store.expand_context(results, window=1)

        # Extract numerical values from results
        number_pattern = re.compile(
            r'[\$€£]?\s*\d[\d,]*\.?\d*\s*(?:billion|million|thousand|%|percent)?',
            re.IGNORECASE,
        )

        extracted = []
        for r in results:
            numbers = number_pattern.findall(r.chunk.text)
            if numbers:
                # Find sentences containing these numbers
                sentences = re.split(r'(?<=[.!?])\s+', r.chunk.text)
                for sent in sentences:
                    sent_numbers = number_pattern.findall(sent)
                    if sent_numbers and len(sent.strip()) > 20:
                        extracted.append({
                            "sentence": sent.strip(),
                            "values": [n.strip() for n in sent_numbers],
                            "section": r.chunk.section_header or "General",
                            "source": r.chunk.source_file,
                            "score": round(r.score, 3),
                        })

        if not extracted:
            return ToolResult(
                tool_name="extract_metrics",
                success=True,
                output=f"No numerical data found for metric: {metric}",
            )

        # Deduplicate by sentence
        seen = set()
        unique = []
        for item in extracted:
            if item["sentence"] not in seen:
                seen.add(item["sentence"])
                unique.append(item)

        lines = [f"Extracted metrics for '{metric}':\n"]
        for item in unique[:15]:
            lines.append(
                f"• [{item['section']}] {item['sentence']}\n"
                f"  Values: {', '.join(item['values'])} (source: {item['source']}, score: {item['score']})"
            )

        return ToolResult(
            tool_name="extract_metrics",
            success=True,
            output="\n".join(lines),
            data={"metrics_found": len(unique)},
        )

    def _calculate(self, params: dict) -> ToolResult:
        expression = params.get("expression", "")
        label = params.get("label", "calculation")

        # Sanitize: only allow numbers, operators, parentheses, whitespace
        sanitized = re.sub(r'[^0-9+\-*/().%\s]', '', expression)
        if not sanitized.strip():
            return ToolResult(
                tool_name="calculate",
                success=False,
                output=f"Invalid expression: {expression}",
            )

        try:
            # Replace % at end with /100 if it looks like "50%"
            sanitized = re.sub(r'(\d)%', r'\1/100', sanitized)
            result = eval(sanitized)  # safe: only numbers and operators after sanitization
            formatted = f"{result:,.4f}" if isinstance(result, float) else str(result)

            return ToolResult(
                tool_name="calculate",
                success=True,
                output=f"{label}: {formatted}\n(Expression: {expression} = {formatted})",
                data={"result": result, "label": label},
            )
        except Exception as e:
            return ToolResult(
                tool_name="calculate",
                success=False,
                output=f"Calculation error: {e}\nExpression was: {expression}",
            )

    def _summarize_findings(self, params: dict) -> ToolResult:
        # This is a signal to the reasoning loop — the actual summarization
        # happens in the LLM. We return a placeholder that the loop uses.
        return ToolResult(
            tool_name="summarize_findings",
            success=True,
            output="[Summarize the findings accumulated in working memory so far.]",
            data={"action": "summarize"},
        )
