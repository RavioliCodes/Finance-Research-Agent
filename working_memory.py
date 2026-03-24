"""
working_memory.py
Scratchpad that accumulates findings across reasoning steps.

The reasoning agent writes to this after each tool call, building up
a structured evidence base that feeds into the final answer.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """A single finding recorded during reasoning."""
    step: int
    tool_used: str
    thought: str
    observation: str
    key_facts: list[str] = field(default_factory=list)


class WorkingMemory:
    """
    Accumulates findings during a multi-step reasoning session.
    Provides the LLM with a structured view of what it has learned so far.
    """

    def __init__(self, question: str):
        self.question = question
        self.entries: list[MemoryEntry] = []
        self.conclusions: list[str] = []
        self.open_questions: list[str] = []
        self.started_at = datetime.now()

    @property
    def step_count(self) -> int:
        return len(self.entries)

    def add_entry(
        self,
        tool_used: str,
        thought: str,
        observation: str,
        key_facts: list[str] | None = None,
    ):
        """Record a reasoning step."""
        self.entries.append(
            MemoryEntry(
                step=self.step_count + 1,
                tool_used=tool_used,
                thought=thought,
                observation=observation,
                key_facts=key_facts or [],
            )
        )

    def add_conclusion(self, conclusion: str):
        """Record a conclusion the agent has reached."""
        self.conclusions.append(conclusion)

    def add_open_question(self, question: str):
        """Record a question that still needs investigation."""
        self.open_questions.append(question)

    def resolve_open_question(self, question: str):
        """Mark an open question as resolved."""
        self.open_questions = [q for q in self.open_questions if q != question]

    def get_summary(self) -> str:
        """
        Render the full working memory as a string for inclusion
        in the LLM prompt. This is what the agent 'sees' about
        its own prior reasoning.
        """
        lines = [f"## Working Memory — {self.step_count} steps so far\n"]
        lines.append(f"**Original Question:** {self.question}\n")

        if self.entries:
            lines.append("### Research Steps:")
            for entry in self.entries:
                lines.append(f"\n**Step {entry.step}** (tool: {entry.tool_used})")
                lines.append(f"  Thought: {entry.thought}")
                lines.append(f"  Observation: {entry.observation[:500]}")
                if entry.key_facts:
                    lines.append("  Key facts extracted:")
                    for fact in entry.key_facts:
                        lines.append(f"    - {fact}")

        if self.conclusions:
            lines.append("\n### Conclusions So Far:")
            for c in self.conclusions:
                lines.append(f"  - {c}")

        if self.open_questions:
            lines.append("\n### Open Questions (still need investigation):")
            for q in self.open_questions:
                lines.append(f"  - {q}")

        return "\n".join(lines)

    def get_all_facts(self) -> list[str]:
        """Collect all key facts across all steps."""
        facts = []
        for entry in self.entries:
            facts.extend(entry.key_facts)
        return facts

    def get_all_observations(self) -> str:
        """Concatenate all observations for final answer generation."""
        return "\n\n".join(
            f"[Step {e.step} — {e.tool_used}]\n{e.observation}"
            for e in self.entries
        )
