"""
strategic_agent.py
Top-level orchestrator that combines the existing LeadershipInsightAgent
(for document ingestion and indexing) with the new ReAct reasoning loop
(for autonomous multi-step research).

Usage:
    from strategic_agent import StrategicAgent

    agent = StrategicAgent(document_folder="./documents")
    agent.initialize()

    # Simple Q&A (uses original single-pass pipeline)
    simple = agent.ask("What is total revenue?")

    # Deep strategic research (uses ReAct reasoning loop)
    deep = agent.research("Should we be concerned about competitive threats
                           given our current market position and R&D investment?")
"""

import os
import time
from agent import LeadershipInsightAgent
from tools import ToolExecutor
from reasoning_agent import ReasoningAgent, ReasoningResult
from answer_generator import AgentResponse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class StrategicAgent:
    """
    Combines document-grounded Q&A with autonomous strategic research.

    - ask(): Single-pass retrieval + answer (fast, for factual questions)
    - research(): Multi-step ReAct reasoning (thorough, for strategic questions)
    """

    def __init__(
        self,
        document_folder: str = "./documents",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        max_reasoning_steps: int = 10,
    ):
        # Reuse the existing agent for document ingestion and indexing
        self.base_agent = LeadershipInsightAgent(
            document_folder=document_folder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            answer_mode="api",  # reasoning loop needs LLM
        )
        self.max_reasoning_steps = max_reasoning_steps
        self._is_ready = False
        self._stats = {}

    def initialize(self) -> dict:
        """Ingest documents and build the search index."""
        self._stats = self.base_agent.initialize()
        self._is_ready = True

        # Wire up the tool executor and reasoning agent
        self._tool_executor = ToolExecutor(self.base_agent)
        self._reasoning_agent = ReasoningAgent(
            tool_executor=self._tool_executor,
            max_steps=self.max_reasoning_steps,
        )

        return self._stats

    def ask(self, question: str, top_k: int = 10) -> AgentResponse:
        """
        Quick single-pass answer (delegates to the original pipeline).
        Best for: factual questions, specific metrics, simple lookups.
        """
        if not self._is_ready:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.base_agent.ask(question, top_k=top_k)

    def research(self, question: str, on_step=None) -> ReasoningResult:
        """
        Deep multi-step research using the ReAct reasoning loop.
        Best for: strategic questions, comparisons, trend analysis,
                  risk assessment, open-ended investigations.

        Args:
            question: The strategic question to investigate.
            on_step: Optional callback(step) for streaming progress updates.

        Returns:
            ReasoningResult with full answer, reasoning chain, and sources.
        """
        if not self._is_ready:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._reasoning_agent.reason(question, on_step=on_step)

    def get_stats(self) -> dict:
        """Return indexing statistics."""
        return {
            **self._stats,
            "mode": "strategic",
            "max_reasoning_steps": self.max_reasoning_steps,
        }

    def get_document_list(self) -> list[str]:
        """Return list of indexed documents."""
        return self.base_agent.get_document_list()


# ── CLI interface ─────────────────────────────────────────────────────────────

def main():
    """Interactive CLI for the Strategic Agent."""
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "./documents"

    agent = StrategicAgent(document_folder=folder)
    stats = agent.initialize()

    print("\nIndexed documents:")
    for doc in agent.get_document_list():
        print(f"  - {doc}")

    print(f"\n{'='*60}")
    print("Strategic Research Agent")
    print("Commands:")
    print("  /ask <question>      — Quick single-pass answer")
    print("  /research <question> — Deep multi-step research")
    print("  /quit                — Exit")
    print(f"{'='*60}\n")

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.lower() in ("/quit", "/exit", "quit", "exit"):
            break

        # Determine mode
        if raw.startswith("/ask "):
            question = raw[5:].strip()
            if not question:
                print("Usage: /ask <question>")
                continue
            t0 = time.time()
            response = agent.ask(question)
            elapsed = time.time() - t0

            print(f"\n{'─'*60}")
            print(f"Mode: Quick Answer | Confidence: {response.confidence.upper()}")
            print(f"{'─'*60}")
            print(f"\n{response.answer}\n")
            print(f"Sources: {', '.join(s['file'] for s in response.sources[:3])}")
            print(f"Time: {elapsed:.2f}s")
            print(f"{'─'*60}\n")

        elif raw.startswith("/research "):
            question = raw[10:].strip()
            if not question:
                print("Usage: /research <question>")
                continue

            print(f"\nResearching: {question}")
            print(f"{'─'*60}")

            def on_step(step):
                icon = "DONE" if step.is_final else f"Step {step.step_number}"
                print(f"\n[{icon}]")
                if step.thought:
                    thought_preview = step.thought[:200]
                    print(f"  Thought: {thought_preview}{'...' if len(step.thought) > 200 else ''}")
                if step.action:
                    print(f"  Action: {step.action}({json.dumps(step.action_input)[:100]})")
                if step.observation and not step.is_final:
                    obs_preview = step.observation[:150]
                    print(f"  Observation: {obs_preview}{'...' if len(step.observation) > 150 else ''}")

            import json
            result = agent.research(question, on_step=on_step)

            print(f"\n{'='*60}")
            print(f"FINAL ANSWER (Confidence: {result.confidence.upper()})")
            print(f"Steps taken: {result.steps_taken} | Time: {result.total_time}s")
            print(f"{'='*60}")
            print(f"\n{result.answer}\n")
            if result.sources:
                print("Sources:", ", ".join(s["file"] for s in result.sources[:5]))
            print(f"{'='*60}\n")

        else:
            # Default to research mode for bare questions
            question = raw

            print(f"\nResearching: {question}")
            print(f"{'─'*60}")

            def on_step(step):
                icon = "DONE" if step.is_final else f"Step {step.step_number}"
                print(f"\n[{icon}]")
                if step.thought:
                    thought_preview = step.thought[:200]
                    print(f"  Thought: {thought_preview}{'...' if len(step.thought) > 200 else ''}")
                if step.action:
                    import json as _json
                    print(f"  Action: {step.action}({_json.dumps(step.action_input)[:100]})")

            result = agent.research(question, on_step=on_step)

            print(f"\n{'='*60}")
            print(f"FINAL ANSWER (Confidence: {result.confidence.upper()})")
            print(f"Steps: {result.steps_taken} | Time: {result.total_time}s")
            print(f"{'='*60}")
            print(f"\n{result.answer}\n")
            print(f"{'='*60}\n")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
