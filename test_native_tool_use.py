"""
test_native_tool_use.py
End-to-end test for native tool use reasoning loop.

Mocks the Anthropic SDK to simulate a multi-step research conversation
without needing API access. Verifies:
- Tool schemas are well-formed
- Reasoning loop handles tool_use / tool_result blocks correctly
- Tool execution dispatches and returns results
- final_answer tool terminates the loop with structured output
- ReasoningStep / ReasoningResult are populated correctly
- on_step callback fires at each step
"""

import sys
import json
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# ── Mock Anthropic SDK response objects ───────────────────────────────────────

@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""

@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}

@dataclass
class MockResponse:
    content: list = None
    stop_reason: str = "end_turn"

    def __post_init__(self):
        if self.content is None:
            self.content = []


# ── Build test fixtures ───────────────────────────────────────────────────────

def make_tool_use_response(thinking: str, tool_name: str, tool_input: dict, tool_id: str = "call_1"):
    """Simulate a response where the model thinks + calls one tool."""
    blocks = []
    if thinking:
        blocks.append(MockTextBlock(text=thinking))
    blocks.append(MockToolUseBlock(id=tool_id, name=tool_name, input=tool_input))
    return MockResponse(content=blocks, stop_reason="tool_use")


def make_final_answer_response(thinking: str, answer: str, confidence: str, tool_id: str = "call_final"):
    """Simulate a response where the model calls final_answer."""
    blocks = []
    if thinking:
        blocks.append(MockTextBlock(text=thinking))
    blocks.append(MockToolUseBlock(
        id=tool_id,
        name="final_answer",
        input={"answer": answer, "confidence": confidence},
    ))
    return MockResponse(content=blocks, stop_reason="tool_use")


def make_text_only_response(text: str):
    """Simulate a response where the model just returns text (no tools)."""
    return MockResponse(
        content=[MockTextBlock(text=text)],
        stop_reason="end_turn",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_tool_schemas():
    """Verify tool schemas are well-formed for both providers."""
    from tools import get_anthropic_tools, get_openai_tools

    anthropic_tools = get_anthropic_tools()
    openai_tools = get_openai_tools()

    assert len(anthropic_tools) == 6
    assert len(openai_tools) == 6

    tool_names = {"search_documents", "compare_sections", "extract_metrics",
                  "calculate", "summarize_findings", "final_answer"}

    # Anthropic format
    for tool in anthropic_tools:
        assert tool["name"] in tool_names, f"Unexpected tool: {tool['name']}"
        assert "description" in tool
        assert "input_schema" in tool
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    # OpenAI format
    for tool in openai_tools:
        assert tool["type"] == "function"
        func = tool["function"]
        assert func["name"] in tool_names
        assert "description" in func
        assert "parameters" in func
        assert func["parameters"]["type"] == "object"

    print("[PASS] test_tool_schemas")


def test_tool_executor():
    """Verify ToolExecutor dispatches correctly with real tool logic."""
    from document_processor import DocumentProcessor
    from vector_store import VectorStore
    from query_expander import QueryExpander
    from agent import LeadershipInsightAgent
    from tools import ToolExecutor

    # Initialize with actual documents
    agent = LeadershipInsightAgent(document_folder="./documents", answer_mode="local")
    agent.initialize()

    executor = ToolExecutor(agent)

    # search_documents
    result = executor.execute("search_documents", {"query": "total revenue"})
    assert result.success, f"search_documents failed: {result.output}"
    assert result.tool_name == "search_documents"
    assert len(result.output) > 0
    print(f"  search_documents: {result.data.get('num_results', 0)} results, top_score={result.data.get('top_score', 0):.3f}")

    # extract_metrics
    result = executor.execute("extract_metrics", {"metric": "revenue"})
    assert result.success, f"extract_metrics failed: {result.output}"
    print(f"  extract_metrics: {result.data.get('metrics_found', 0)} metrics found")

    # calculate
    result = executor.execute("calculate", {"expression": "(19.06 - 17.61) / 17.61 * 100", "label": "YoY growth %"})
    assert result.success, f"calculate failed: {result.output}"
    assert "8.23" in result.output  # ~8.23%
    print(f"  calculate: {result.output.splitlines()[0]}")

    # compare_sections
    result = executor.execute("compare_sections", {"query_a": "revenue 2024", "query_b": "revenue 2023"})
    assert result.success, f"compare_sections failed: {result.output}"
    print(f"  compare_sections: a={result.data.get('results_a_count', 0)}, b={result.data.get('results_b_count', 0)}")

    # summarize_findings
    result = executor.execute("summarize_findings", {})
    assert result.success

    # unknown tool
    result = executor.execute("nonexistent_tool", {})
    assert not result.success
    assert "Unknown tool" in result.output

    # final_answer is NOT handled by ToolExecutor (handled by reasoning loop)
    result = executor.execute("final_answer", {"answer": "test", "confidence": "high"})
    assert not result.success  # ToolExecutor doesn't know about final_answer

    print("[PASS] test_tool_executor")


def test_reasoning_loop_anthropic():
    """
    Simulate a 3-step Anthropic native tool use conversation:
    1. Model calls search_documents → gets results
    2. Model calls extract_metrics → gets metrics
    3. Model calls final_answer → loop terminates with structured answer
    """
    from agent import LeadershipInsightAgent
    from tools import ToolExecutor
    from reasoning_agent import ReasoningAgent, ReasoningStep, ReasoningResult

    # Set up real agent + tool executor (for real tool execution)
    base_agent = LeadershipInsightAgent(document_folder="./documents", answer_mode="local")
    base_agent.initialize()
    executor = ToolExecutor(base_agent)

    agent = ReasoningAgent(tool_executor=executor, max_steps=10, provider="anthropic")

    # Define the sequence of mock LLM responses
    mock_responses = [
        # Turn 1: search for revenue data
        make_tool_use_response(
            thinking="I need to find revenue data. Let me search the documents.",
            tool_name="search_documents",
            tool_input={"query": "total revenue fiscal year", "top_k": 5},
            tool_id="call_001",
        ),
        # Turn 2: extract specific metrics
        make_tool_use_response(
            thinking="I found some revenue references. Let me extract the exact numbers.",
            tool_name="extract_metrics",
            tool_input={"metric": "total revenue"},
            tool_id="call_002",
        ),
        # Turn 3: final answer
        make_final_answer_response(
            thinking="I now have enough data to provide a comprehensive answer.",
            answer=(
                "Based on the financial documents, Adobe's total revenue for fiscal year 2025 "
                "was $21.50 billion, representing year-over-year growth. The company's revenue "
                "is driven primarily by its Digital Media and Digital Experience segments."
            ),
            confidence="high",
            tool_id="call_003",
        ),
    ]

    # Track on_step callbacks
    callback_steps = []
    def on_step(step):
        callback_steps.append(step)
        icon = "FINAL" if step.is_final else f"Step {step.step_number}"
        action_str = f" → {step.action}" if step.action else ""
        print(f"  [{icon}]{action_str}")
        if step.thought:
            print(f"    Thought: {step.thought[:80]}...")
        if step.observation and not step.is_final:
            print(f"    Observation: {step.observation[:80]}...")

    # Mock the anthropic SDK
    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(side_effect=mock_responses)

    with patch("anthropic.Anthropic", return_value=mock_client):
        # Need to set the API key so it takes the Anthropic path
        agent.anthropic_api_key = "sk-test-mock"
        agent.provider = "anthropic"

        result = agent.reason("What is Adobe's total revenue?", on_step=on_step)

    # ── Assertions ────────────────────────────────────────────────────────
    assert isinstance(result, ReasoningResult)
    assert result.question == "What is Adobe's total revenue?"
    assert "21.50 billion" in result.answer
    assert result.confidence == "high"
    assert result.steps_taken == 3  # search + extract + final_answer
    assert result.total_time >= 0
    assert len(result.reasoning_steps) == 3

    # Step 1: search_documents
    s1 = result.reasoning_steps[0]
    assert s1.action == "search_documents"
    assert s1.action_input == {"query": "total revenue fiscal year", "top_k": 5}
    assert not s1.is_final
    assert len(s1.observation) > 0  # Real tool output

    # Step 2: extract_metrics
    s2 = result.reasoning_steps[1]
    assert s2.action == "extract_metrics"
    assert not s2.is_final
    assert len(s2.observation) > 0

    # Step 3: final_answer
    s3 = result.reasoning_steps[2]
    assert s3.action == "final_answer"
    assert s3.is_final
    assert "21.50 billion" in s3.action_input["answer"]

    # Callbacks fired for each step
    assert len(callback_steps) == 3
    assert callback_steps[0].action == "search_documents"
    assert callback_steps[1].action == "extract_metrics"
    assert callback_steps[2].is_final

    # Sources extracted from working memory
    assert isinstance(result.sources, list)

    # Verify the API was called 3 times (one per turn)
    calls = mock_client.messages.create.call_args_list
    assert len(calls) == 3

    # All calls should include 6 tools
    for call in calls:
        assert len(call.kwargs["tools"]) == 6

    # The final messages list (shared by reference) should have the full conversation:
    # user, assistant(tool_use), user(tool_result),
    # assistant(tool_use), user(tool_result),
    # assistant(final_answer)
    final_msgs = calls[-1].kwargs["messages"]
    assert final_msgs[0]["role"] == "user"
    assert final_msgs[0]["content"] == "What is Adobe's total revenue?"

    # Check alternating assistant/user pattern with tool_use/tool_result
    assert final_msgs[1]["role"] == "assistant"
    assert any(b["type"] == "tool_use" for b in final_msgs[1]["content"])
    assert final_msgs[2]["role"] == "user"
    assert final_msgs[2]["content"][0]["type"] == "tool_result"
    assert final_msgs[2]["content"][0]["tool_use_id"] == "call_001"

    print("[PASS] test_reasoning_loop_anthropic")


def test_reasoning_loop_text_fallback():
    """
    Test that the loop handles the model returning plain text (no tools)
    as a direct answer — the end_turn case.
    """
    from agent import LeadershipInsightAgent
    from tools import ToolExecutor
    from reasoning_agent import ReasoningAgent

    base_agent = LeadershipInsightAgent(document_folder="./documents", answer_mode="local")
    base_agent.initialize()
    executor = ToolExecutor(base_agent)

    agent = ReasoningAgent(tool_executor=executor, max_steps=10, provider="anthropic")

    # Model just returns text immediately without calling any tools
    mock_responses = [
        make_text_only_response(
            "I could not find enough information to answer this question. "
            "Please provide more specific financial documents."
        ),
    ]

    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(side_effect=mock_responses)

    with patch("anthropic.Anthropic", return_value=mock_client):
        agent.anthropic_api_key = "sk-test-mock"
        agent.provider = "anthropic"
        result = agent.reason("What is the meaning of life?")

    assert "could not find enough information" in result.answer
    assert result.confidence == "low"  # inferred from "unable" language
    assert result.steps_taken == 1
    assert result.reasoning_steps[0].is_final
    assert result.reasoning_steps[0].action is None

    print("[PASS] test_reasoning_loop_text_fallback")


def test_reasoning_loop_multi_tool_single_turn():
    """
    Test that the loop correctly handles multiple tool calls in a single
    API response (parallel tool use).
    """
    from agent import LeadershipInsightAgent
    from tools import ToolExecutor
    from reasoning_agent import ReasoningAgent

    base_agent = LeadershipInsightAgent(document_folder="./documents", answer_mode="local")
    base_agent.initialize()
    executor = ToolExecutor(base_agent)

    agent = ReasoningAgent(tool_executor=executor, max_steps=10, provider="anthropic")

    # Turn 1: model calls TWO tools at once (parallel tool use)
    turn1 = MockResponse(
        content=[
            MockTextBlock(text="I'll search for both revenue and expenses simultaneously."),
            MockToolUseBlock(id="call_a", name="search_documents", input={"query": "total revenue"}),
            MockToolUseBlock(id="call_b", name="search_documents", input={"query": "operating expenses"}),
        ],
        stop_reason="tool_use",
    )
    # Turn 2: final answer
    turn2 = make_final_answer_response(
        thinking="Now I can compare.",
        answer="Revenue exceeds operating expenses, indicating profitability.",
        confidence="medium",
    )

    callback_steps = []
    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(side_effect=[turn1, turn2])

    with patch("anthropic.Anthropic", return_value=mock_client):
        agent.anthropic_api_key = "sk-test-mock"
        agent.provider = "anthropic"
        result = agent.reason("Compare revenue and expenses", on_step=lambda s: callback_steps.append(s))

    # 2 tool calls from turn 1 + 1 final_answer from turn 2 = 3 steps
    assert result.steps_taken == 3
    assert len(callback_steps) == 3
    assert callback_steps[0].action == "search_documents"
    assert callback_steps[1].action == "search_documents"
    assert callback_steps[2].action == "final_answer"
    assert callback_steps[2].is_final

    # Verify both tool_results were sent back in one user message
    # (messages list is shared by reference, so check by index in final state)
    final_msgs = mock_client.messages.create.call_args_list[-1].kwargs["messages"]
    # Index 0: user question, 1: assistant(2 tool_use), 2: user(2 tool_result), 3: assistant(final)
    tool_result_msg = final_msgs[2]
    assert tool_result_msg["role"] == "user"
    assert len(tool_result_msg["content"]) == 2  # Two tool_result blocks
    ids = {tr["tool_use_id"] for tr in tool_result_msg["content"]}
    assert ids == {"call_a", "call_b"}

    print("[PASS] test_reasoning_loop_multi_tool_single_turn")


def test_reasoning_loop_api_error():
    """Test graceful fallback when the API call fails."""
    from agent import LeadershipInsightAgent
    from tools import ToolExecutor
    from reasoning_agent import ReasoningAgent

    base_agent = LeadershipInsightAgent(document_folder="./documents", answer_mode="local")
    base_agent.initialize()
    executor = ToolExecutor(base_agent)

    agent = ReasoningAgent(tool_executor=executor, max_steps=10, provider="anthropic")

    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(side_effect=Exception("Connection timeout"))

    with patch("anthropic.Anthropic", return_value=mock_client):
        agent.anthropic_api_key = "sk-test-mock"
        agent.provider = "anthropic"
        result = agent.reason("What is revenue?")

    assert result.confidence == "low"
    assert "unable to complete" in result.answer.lower() or "error" in result.answer.lower()
    assert result.steps_taken == 0

    print("[PASS] test_reasoning_loop_api_error")


def test_max_steps_enforcement():
    """Test that the loop forces final_answer when max_steps is reached."""
    from agent import LeadershipInsightAgent
    from tools import ToolExecutor
    from reasoning_agent import ReasoningAgent

    base_agent = LeadershipInsightAgent(document_folder="./documents", answer_mode="local")
    base_agent.initialize()
    executor = ToolExecutor(base_agent)

    agent = ReasoningAgent(tool_executor=executor, max_steps=2, provider="anthropic")

    # Model keeps calling tools, never calls final_answer
    responses = [
        make_tool_use_response("Searching...", "search_documents", {"query": "revenue"}, "c1"),
        make_tool_use_response("More searching...", "search_documents", {"query": "expenses"}, "c2"),
        # After max_steps, the loop adds a "you must call final_answer" message
        # and the model complies:
        make_final_answer_response("OK wrapping up.", "Revenue is $21B.", "medium", "c3"),
    ]

    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(side_effect=responses)

    with patch("anthropic.Anthropic", return_value=mock_client):
        agent.anthropic_api_key = "sk-test-mock"
        agent.provider = "anthropic"
        result = agent.reason("What is revenue?")

    assert result.answer == "Revenue is $21B."
    assert result.confidence == "medium"

    # Verify the forced message was injected (messages list is shared by reference,
    # so we check the final state which includes all messages)
    all_msgs = mock_client.messages.create.call_args_list[-1].kwargs["messages"]
    forced_msgs = [m for m in all_msgs if isinstance(m.get("content"), str) and "MUST" in m.get("content", "")]
    assert len(forced_msgs) >= 1, "Expected a forced final_answer message"

    print("[PASS] test_max_steps_enforcement")


def test_message_structure():
    """Verify the exact message structure sent to the Anthropic API."""
    from agent import LeadershipInsightAgent
    from tools import ToolExecutor
    from reasoning_agent import ReasoningAgent

    base_agent = LeadershipInsightAgent(document_folder="./documents", answer_mode="local")
    base_agent.initialize()
    executor = ToolExecutor(base_agent)

    agent = ReasoningAgent(tool_executor=executor, max_steps=10, provider="anthropic")

    responses = [
        make_tool_use_response("Let me calculate.", "calculate",
                               {"expression": "100 / 4", "label": "test"}, "c1"),
        make_final_answer_response("Done.", "The answer is 25.", "high", "c2"),
    ]

    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(side_effect=responses)

    with patch("anthropic.Anthropic", return_value=mock_client):
        agent.anthropic_api_key = "sk-test-mock"
        agent.provider = "anthropic"
        agent.reason("What is 100/4?")

    # Inspect final messages list (shared by reference across all calls)
    msgs = mock_client.messages.create.call_args_list[-1].kwargs["messages"]

    # Message 0: user question
    assert msgs[0] == {"role": "user", "content": "What is 100/4?"}

    # Message 1: assistant with text + tool_use
    assert msgs[1]["role"] == "assistant"
    content = msgs[1]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Let me calculate."}
    assert content[1]["type"] == "tool_use"
    assert content[1]["id"] == "c1"
    assert content[1]["name"] == "calculate"
    assert content[1]["input"] == {"expression": "100 / 4", "label": "test"}

    # Message 2: user with tool_result
    assert msgs[2]["role"] == "user"
    tool_results = msgs[2]["content"]
    assert len(tool_results) == 1
    assert tool_results[0]["type"] == "tool_result"
    assert tool_results[0]["tool_use_id"] == "c1"
    assert "25" in tool_results[0]["content"]  # calculate result

    print("[PASS] test_message_structure")


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Native Tool Use — End-to-End Tests")
    print("=" * 60)

    print("\n1. Tool schema validation")
    test_tool_schemas()

    print("\n2. Tool executor dispatch (real documents)")
    test_tool_executor()

    print("\n3. Full reasoning loop — 3-step Anthropic conversation")
    test_reasoning_loop_anthropic()

    print("\n4. Text-only fallback (end_turn, no tools)")
    test_reasoning_loop_text_fallback()

    print("\n5. Parallel tool use (multiple tools in one turn)")
    test_reasoning_loop_multi_tool_single_turn()

    print("\n6. API error graceful fallback")
    test_reasoning_loop_api_error()

    print("\n7. Max steps enforcement")
    test_max_steps_enforcement()

    print("\n8. Message structure verification")
    test_message_structure()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
