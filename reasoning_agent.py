"""
reasoning_agent.py
Autonomous reasoning loop using native tool use.

Uses Anthropic's native tool use (structured tool_use / tool_result blocks)
instead of text-based ReAct parsing. Falls back to OpenAI's function calling
when configured.
"""

import os
import re
import json
import time
from dataclasses import dataclass, field
from working_memory import WorkingMemory
from tools import ToolExecutor, ToolResult, get_anthropic_tools, get_openai_tools

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_number: int
    thought: str = ""
    action: str | None = None
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False


@dataclass
class ReasoningResult:
    """The full result of a reasoning session."""
    question: str
    answer: str
    confidence: str
    reasoning_steps: list[ReasoningStep]
    sources: list[dict]
    total_time: float
    steps_taken: int


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an autonomous Financial Research Agent that answers complex, open-ended \
strategic questions by iteratively researching company financial documents.

Use the available tools to search documents, extract metrics, compare data across \
time periods or segments, and perform calculations. Build your understanding step \
by step before providing a final answer.

## Guidelines

- Break complex questions into sub-questions and investigate each one.
- Always search for SPECIFIC metrics — don't ask vague queries.
- Use extract_metrics when you need precise numbers.
- Use compare_sections for year-over-year or segment comparisons.
- Use calculate to compute ratios, growth rates, or margins from extracted data.
- Use summarize_findings when you have accumulated several observations \
and need to organize before continuing.
- Aim for 3-8 tool calls. Use fewer for simple questions, more for complex ones.
- When you have gathered enough evidence, call final_answer with a comprehensive, \
evidence-based answer. Cite specific numbers, time periods, and document sections.
- If you cannot find enough information, call final_answer explaining what's missing.
- Do NOT exceed {max_steps} tool calls — call final_answer with the best evidence you have.\
"""


class ReasoningAgent:
    """
    Runs an autonomous research loop powered by native tool use.
    Supports Anthropic (via SDK) and OpenAI (via requests) as LLM providers.
    """

    def __init__(
        self,
        tool_executor: ToolExecutor,
        max_steps: int = 10,
        provider: str | None = None,
    ):
        self.tool_executor = tool_executor
        self.max_steps = max_steps

        # LLM config
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_api_url = os.environ.get(
            "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
        )

    def reason(self, question: str, on_step=None) -> ReasoningResult:
        """
        Run the full reasoning loop for a question.

        Args:
            question: The strategic question to answer.
            on_step: Optional callback(step: ReasoningStep) for streaming progress.

        Returns:
            ReasoningResult with the final answer, reasoning chain, and metadata.
        """
        if self.provider == "anthropic" and self.anthropic_api_key:
            return self._reason_anthropic(question, on_step)
        if self.openai_api_key:
            return self._reason_openai(question, on_step)
        raise RuntimeError(
            "No LLM API key configured. The reasoning agent requires an API key. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
        )

    # ── Anthropic native tool use ─────────────────────────────────────────────

    def _reason_anthropic(self, question: str, on_step=None) -> ReasoningResult:
        """Run reasoning loop using Anthropic's native tool use via the SDK."""
        import anthropic

        t0 = time.time()
        memory = WorkingMemory(question)
        steps: list[ReasoningStep] = []
        step_counter = 0

        client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        tools = get_anthropic_tools()
        system = SYSTEM_PROMPT.format(max_steps=self.max_steps)

        messages = [{"role": "user", "content": question}]

        for _turn in range(self.max_steps + 2):  # +2 for forced final answer turns
            try:
                response = client.messages.create(
                    model=self.anthropic_model,
                    max_tokens=4096,
                    system=system,
                    tools=tools,
                    messages=messages,
                    temperature=0.2,
                )
            except Exception as e:
                print(f"[ReasoningAgent] Anthropic call failed: {e}")
                return self._build_fallback_result(question, memory, steps, time.time() - t0)

            # Serialize assistant content for message history
            assistant_content = []
            thinking_text = ""
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                    thinking_text += block.text
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            # Get tool_use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            # ── No tool calls → model produced a final text answer ────────
            if not tool_use_blocks:
                step_counter += 1
                step = ReasoningStep(
                    step_number=step_counter,
                    thought=thinking_text,
                    is_final=True,
                )
                steps.append(step)
                if on_step:
                    on_step(step)

                return ReasoningResult(
                    question=question,
                    answer=thinking_text,
                    confidence=self._infer_confidence(thinking_text),
                    reasoning_steps=steps,
                    sources=self._collect_sources(memory),
                    total_time=round(time.time() - t0, 2),
                    steps_taken=step_counter,
                )

            # ── Check for final_answer tool ───────────────────────────────
            final_block = next(
                (b for b in tool_use_blocks if b.name == "final_answer"), None
            )
            if final_block:
                answer_text = final_block.input.get("answer", thinking_text)
                confidence = final_block.input.get("confidence", "medium")

                step_counter += 1
                step = ReasoningStep(
                    step_number=step_counter,
                    thought=thinking_text if thinking_text else answer_text,
                    action="final_answer",
                    action_input=final_block.input,
                    observation=answer_text,
                    is_final=True,
                )
                steps.append(step)
                if on_step:
                    on_step(step)

                return ReasoningResult(
                    question=question,
                    answer=answer_text,
                    confidence=confidence,
                    reasoning_steps=steps,
                    sources=self._collect_sources(memory),
                    total_time=round(time.time() - t0, 2),
                    steps_taken=step_counter,
                )

            # ── Execute tools and collect results ─────────────────────────
            tool_results_for_api = []

            for block in tool_use_blocks:
                result = self.tool_executor.execute(block.name, block.input)

                step_counter += 1
                step = ReasoningStep(
                    step_number=step_counter,
                    thought=thinking_text,
                    action=block.name,
                    action_input=block.input,
                    observation=result.output,
                )
                steps.append(step)
                if on_step:
                    on_step(step)

                # Update working memory
                memory.add_entry(
                    tool_used=block.name,
                    thought=thinking_text,
                    observation=result.output[:1000],
                    key_facts=self._extract_key_facts(result.output),
                )

                tool_results_for_api.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.output,
                })

            messages.append({"role": "user", "content": tool_results_for_api})

            # Check if we've hit the step limit — force final answer
            if step_counter >= self.max_steps:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have reached the maximum of {self.max_steps} research steps. "
                        f"You MUST now call the final_answer tool with your best answer "
                        f"based on all evidence gathered so far."
                    ),
                })

        # Exhausted all turns — build from memory
        return self._build_fallback_result(question, memory, steps, time.time() - t0)

    # ── OpenAI native function calling ────────────────────────────────────────

    def _reason_openai(self, question: str, on_step=None) -> ReasoningResult:
        """Run reasoning loop using OpenAI's native function calling via requests."""
        import requests as req

        t0 = time.time()
        memory = WorkingMemory(question)
        steps: list[ReasoningStep] = []
        step_counter = 0

        tools = get_openai_tools()
        system = SYSTEM_PROMPT.format(max_steps=self.max_steps)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        for _turn in range(self.max_steps + 2):
            try:
                resp = req.post(
                    self.openai_api_url,
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.openai_model,
                        "messages": messages,
                        "tools": tools,
                        "temperature": 0.2,
                        "max_tokens": 4096,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"[ReasoningAgent] OpenAI call failed: {e}")
                return self._build_fallback_result(question, memory, steps, time.time() - t0)

            choice = data["choices"][0]
            message = choice["message"]

            # Append the full assistant message to history
            messages.append(message)

            thinking_text = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            # ── No tool calls → final text answer ────────────────────────
            if not tool_calls:
                step_counter += 1
                step = ReasoningStep(
                    step_number=step_counter,
                    thought=thinking_text,
                    is_final=True,
                )
                steps.append(step)
                if on_step:
                    on_step(step)

                return ReasoningResult(
                    question=question,
                    answer=thinking_text,
                    confidence=self._infer_confidence(thinking_text),
                    reasoning_steps=steps,
                    sources=self._collect_sources(memory),
                    total_time=round(time.time() - t0, 2),
                    steps_taken=step_counter,
                )

            # ── Check for final_answer ────────────────────────────────────
            final_tc = next(
                (tc for tc in tool_calls if tc["function"]["name"] == "final_answer"),
                None,
            )
            if final_tc:
                try:
                    args = json.loads(final_tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {"answer": thinking_text, "confidence": "medium"}

                answer_text = args.get("answer", thinking_text)
                confidence = args.get("confidence", "medium")

                step_counter += 1
                step = ReasoningStep(
                    step_number=step_counter,
                    thought=thinking_text if thinking_text else answer_text,
                    action="final_answer",
                    action_input=args,
                    observation=answer_text,
                    is_final=True,
                )
                steps.append(step)
                if on_step:
                    on_step(step)

                return ReasoningResult(
                    question=question,
                    answer=answer_text,
                    confidence=confidence,
                    reasoning_steps=steps,
                    sources=self._collect_sources(memory),
                    total_time=round(time.time() - t0, 2),
                    steps_taken=step_counter,
                )

            # ── Execute tools ─────────────────────────────────────────────
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                result = self.tool_executor.execute(func_name, func_args)

                step_counter += 1
                step = ReasoningStep(
                    step_number=step_counter,
                    thought=thinking_text,
                    action=func_name,
                    action_input=func_args,
                    observation=result.output,
                )
                steps.append(step)
                if on_step:
                    on_step(step)

                memory.add_entry(
                    tool_used=func_name,
                    thought=thinking_text,
                    observation=result.output[:1000],
                    key_facts=self._extract_key_facts(result.output),
                )

                # OpenAI expects one tool message per tool call
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result.output,
                })

            # Check step limit
            if step_counter >= self.max_steps:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have reached the maximum of {self.max_steps} research steps. "
                        f"You MUST now call the final_answer tool with your best answer "
                        f"based on all evidence gathered so far."
                    ),
                })

        return self._build_fallback_result(question, memory, steps, time.time() - t0)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _extract_key_facts(self, observation: str) -> list[str]:
        """Extract notable facts from a tool observation for working memory."""
        facts = []
        sentences = re.split(r'(?<=[.!?])\s+', observation)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 300:
                continue
            if re.search(r'[\$€£]\s*\d|%|\d+\.\d+\s*(?:billion|million)', sent, re.IGNORECASE):
                facts.append(sent)
                if len(facts) >= 5:
                    break
        return facts

    def _collect_sources(self, memory: WorkingMemory) -> list[dict]:
        """Collect source references from working memory entries."""
        sources = []
        seen = set()
        for entry in memory.entries:
            for match in re.finditer(r'(\S+\.pdf|\S+\.txt|\S+\.md)', entry.observation):
                src = match.group(1).strip('[]|')
                if src not in seen:
                    seen.add(src)
                    sources.append({"file": src, "used_in_step": entry.step})
        return sources

    def _infer_confidence(self, text: str) -> str:
        """Infer confidence from the final text when final_answer tool wasn't used."""
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in [
            "insufficient", "could not find", "no relevant", "unable to",
            "not enough information", "missing",
        ]):
            return "low"
        if any(phrase in text_lower for phrase in [
            "clearly", "strong evidence", "well-documented", "confirms",
        ]):
            return "high"
        return "medium"

    def _build_fallback_result(
        self, question: str, memory: WorkingMemory,
        steps: list[ReasoningStep], elapsed: float,
    ) -> ReasoningResult:
        """Build a result when the LLM call fails or max turns exhausted."""
        if memory.entries:
            answer = (
                "I encountered an error during research, but here is what I found so far:\n\n"
                + "\n".join(f"- {f}" for f in memory.get_all_facts())
            )
        else:
            answer = (
                "I was unable to complete the research due to an API error. "
                "Please check your API key configuration."
            )

        return ReasoningResult(
            question=question,
            answer=answer,
            confidence="low",
            reasoning_steps=steps,
            sources=self._collect_sources(memory),
            total_time=round(elapsed, 2),
            steps_taken=len(steps),
        )
