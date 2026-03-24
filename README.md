# AI Leadership Insight & Decision Agent

An AI-powered assistant for organizational leadership that answers strategic questions grounded in internal company documents. Supports both **quick single-pass Q&A** and **autonomous multi-step research** for complex strategic analysis.

---

## Architecture

### V1 — Quick Q&A Pipeline

```
┌─────────────────────────────────────────────────────┐
│                  Web UI (Flask)                      │
│              Leadership Question Input                │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Query Expander  │  Synonym expansion, acronym
              │                  │  resolution, decomposition
              └────────┬────────┘
                       │  [multiple query variations]
              ┌────────▼────────┐
              │  Vector Store    │  TF-IDF indexing + cosine
              │  (Retriever)     │  similarity + RRF fusion
              └────────┬────────┘
                       │  [top-k relevant chunks]
              ┌────────▼────────┐
              │ Answer Generator │  Extractive (local) or
              │                  │  LLM API-based generation
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Structured      │  Answer + confidence +
              │  Response        │  sources + metadata
              └─────────────────┘
```

### V2 — Autonomous Research Pipeline (Native Tool Use)

```
┌──────────────────────────────────────────────────────────────┐
│                    Web UI / CLI                               │
│             Strategic Question Input                          │
└──────────────────────┬───────────────────────────────────────┘
                       │
              ┌────────▼─────────┐
              │  Strategic Agent  │  Routes to quick or research mode
              └────────┬─────────┘
                       │
          ┌────────────▼────────────────┐
          │    Native Tool Use Loop      │
          │  (Anthropic / OpenAI SDK)    │
          │                              │
          │  ┌────────────────────────┐  │
          │  │     Working Memory     │  │
          │  │  (accumulated evidence) │  │
          │  └────────────────────────┘  │
          │                              │
          │  ┌────────────────────────┐  │
          │  │     Tool Executor      │  │
          │  │  • search_documents    │  │
          │  │  • compare_sections    │  │
          │  │  • extract_metrics     │  │
          │  │  • calculate           │  │
          │  │  • summarize_findings  │  │
          │  │  • final_answer        │  │
          │  └────────────────────────┘  │
          │                              │
          │  LLM returns structured      │
          │  tool_use blocks; repeats    │
          │  until final_answer called   │
          └────────────┬────────────────┘
                       │
              ┌────────▼────────┐
              │  Final Answer    │  Comprehensive analysis +
              │                  │  confidence + sources +
              │                  │  full reasoning chain
              └─────────────────┘
```

**Example research session:**
```
Question: "Should we be concerned about competitive threats
           given our R&D spending trends?"

Turn 1: LLM calls  search_documents({"query": "R&D expenditure fiscal 2024"})
        → Tool returns 8 relevant excerpts with scores

Turn 2: LLM calls  search_documents({"query": "competition market position threats"})
        → Tool returns competitive landscape excerpts

Turn 3: LLM calls  compare_sections({"query_a": "R&D fiscal 2024", "query_b": "R&D fiscal 2023"})
        → Tool returns side-by-side comparison

Turn 4: LLM calls  calculate({"expression": "(3.14 - 2.99) / 2.99 * 100", "label": "R&D YoY growth"})
        → Tool returns: 5.0167%

Turn 5: LLM calls  final_answer({"answer": "...", "confidence": "high"})
        → Comprehensive strategic analysis with citations
```

---

## Components

### Core (V1 — Document Q&A)

| File | Purpose |
|------|---------|
| `document_processor.py` | Ingests documents, detects sections, creates overlapping chunks |
| `vector_store.py` | TF-IDF vectorization, cosine similarity search, RRF multi-query fusion |
| `query_expander.py` | Expands queries with synonyms, acronyms, and decomposition |
| `answer_generator.py` | Generates grounded answers (local extractive or LLM API) |
| `agent.py` | Main orchestrator + CLI interface |
| `app.py` | Flask web server with REST API (port 5000) |
| `demo.py` | Demonstration script with sample questions |
| `static/index.html` | Web UI for quick Q&A |

### Autonomous Research (V2 — Strategic Agent)

| File | Purpose |
|------|---------|
| `tools.py` | 6 tools with JSON Schema definitions for native tool use: `search_documents`, `compare_sections`, `extract_metrics`, `calculate`, `summarize_findings`, `final_answer` |
| `working_memory.py` | Scratchpad that accumulates findings across reasoning steps |
| `reasoning_agent.py` | Autonomous reasoning loop using Anthropic native tool use (SDK) / OpenAI function calling |
| `strategic_agent.py` | Top-level orchestrator with both `ask()` (quick) and `research()` (deep) modes |
| `app_v2.py` | Flask web server with `/api/research` endpoint (port 5001) |
| `static/index_v2.html` | Updated UI with Quick/Research mode toggle and reasoning step viewer |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Documents
Place company documents (`.txt`, `.md`, `.csv`, `.pdf`) in the `documents/` folder. Sample documents are included for demonstration.

### 3. Configure API Keys (required for Research mode)
Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
# Edit .env with your keys
```

Research mode requires an LLM API key (Anthropic or OpenAI). Quick Q&A mode works without any API key using local extractive generation.

### 4. Start the App

**V1 — Quick Q&A only:**
```bash
python app.py
```
Then open http://localhost:5000

**V2 — Quick Q&A + Autonomous Research:**
```bash
python app_v2.py
```
Then open http://localhost:5001

### 5. Interactive CLI

**V1 CLI:**
```bash
python agent.py ./documents
```

**V2 CLI (with research mode):**
```bash
python strategic_agent.py ./documents
```
CLI commands:
- `/ask <question>` — Quick single-pass answer
- `/research <question>` — Deep multi-step research
- Type a bare question — defaults to research mode

### Optional: Run the Demo
```bash
python demo.py
```

---

## API Endpoints

### V1 (app.py — port 5000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ask` | Ask a question. Body: `{"question": "..."}` |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/upload` | Upload a new document (multipart form) |
| `GET` | `/api/stats` | Get index statistics |

### V2 (app_v2.py — port 5001)

All V1 endpoints plus:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/research` | Deep multi-step research. Body: `{"question": "..."}` |

**Research response format:**
```json
{
  "answer": "Comprehensive strategic analysis...",
  "confidence": "high",
  "sources": [{"file": "10k.pdf", "used_in_step": 1}],
  "reasoning_steps": [
    {
      "step": 1,
      "thought": "I need to find revenue data...",
      "action": "search_documents",
      "action_input": {"query": "total revenue fiscal 2024"},
      "observation": "Found 8 relevant excerpts...",
      "is_final": false
    }
  ],
  "steps_taken": 5,
  "response_time": 12.34,
  "mode": "research"
}
```

---

## Answer Modes

### Local Mode (Default for V1)
Runs entirely offline using TF-IDF retrieval and extractive answer generation. No API keys needed.

### API Mode
Set environment variables to use an LLM API (or add them to a `.env` file):

**OpenAI (default provider):**
```bash
export ANSWER_MODE=api
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL=gpt-4o-mini
```

**Anthropic:**
```bash
export ANSWER_MODE=api
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-api-key
export ANTHROPIC_MODEL=claude-sonnet-4-6
```

### Research Mode (V2)
Always uses an LLM API for the autonomous reasoning loop via native tool use. Falls back gracefully if API is unavailable.

---

## Research Agent Tools

The autonomous reasoning agent has 6 tools, defined as JSON Schema and served via native tool use (Anthropic `tool_use` blocks / OpenAI function calling):

| Tool | Description |
|------|-------------|
| `search_documents` | Search indexed documents for specific facts, figures, or qualitative statements |
| `compare_sections` | Search two topics/periods side-by-side for comparison analysis |
| `extract_metrics` | Extract all numerical values for a specific financial metric |
| `calculate` | Perform arithmetic — ratios, growth rates, margins from extracted data |
| `summarize_findings` | Organize accumulated evidence before forming a final answer |
| `final_answer` | Structured termination — returns the answer with an explicit confidence level (`high`/`medium`/`low`) |

---

## Key Design Decisions

- **TF-IDF over dense embeddings**: Works offline, no API costs, fast indexing. Could be later transformed to sentence-transformers + FAISS.
- **Multi-query retrieval with RRF**: Query expansion + Reciprocal Rank Fusion improves recall significantly over single-query search.
- **Section-aware chunking**: Respects document structure (headers, sections) for better context preservation.
- **Confidence scoring**: Based on retrieval scores to help leadership gauge answer reliability.
- **Native tool use over text-based ReAct**: The reasoning loop uses Anthropic's native `tool_use` / `tool_result` blocks (and OpenAI's function calling) instead of parsing free-text `Thought → Action → Observation` patterns. This gives structured, typed tool calls with no regex parsing and eliminates malformed-JSON failures.
- **Working memory**: Accumulates facts across steps so the agent builds a coherent evidence base before answering.
- **No framework dependency**: The agent loop is built from scratch — no LangChain/LlamaIndex — keeping the codebase simple and transparent.

## Sample Questions

**Quick mode:**
- "What is our current revenue trend?"
- "Which departments are underperforming?"
- "What were the key risks highlighted in the last quarter?"

**Research mode (strategic, open-ended):**
- "Should we be concerned about our competitive position given current R&D investment levels?"
- "What is the relationship between our subscription revenue growth and overall margin trajectory?"
- "Analyze the key risk factors and assess which ones pose the greatest near-term threat to profitability."
- "Compare our capital allocation strategy across segments — where are we investing most and is it paying off?"
