"""
app_v2.py
Flask web application for the Strategic Research Agent.

Adds the /api/research endpoint for multi-step autonomous reasoning
while keeping the original /api/ask endpoint for quick answers.
"""

import os
import json
import time
import threading
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

from strategic_agent import StrategicAgent

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "documents")

AGENT = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_agent():
    global AGENT
    if AGENT is None:
        doc_folder = app.config["UPLOAD_FOLDER"]
        os.makedirs(doc_folder, exist_ok=True)
        AGENT = StrategicAgent(document_folder=doc_folder)
        AGENT.initialize()
    return AGENT


# ── Static files ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("static/index_v2.html")


# ── Quick ask (original pipeline) ────────────────────────────────────────────

@app.route("/api/ask", methods=["POST"])
def ask_question():
    """Quick single-pass answer."""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    agent = get_agent()
    t0 = time.time()
    response = agent.ask(question, top_k=10)
    elapsed = time.time() - t0

    return jsonify({
        "answer": response.answer,
        "confidence": response.confidence,
        "sources": response.sources,
        "query_variations": response.query_variations,
        "response_time": round(elapsed, 3),
        "mode": "quick",
    })


# ── Deep research (new ReAct pipeline) ───────────────────────────────────────

@app.route("/api/research", methods=["POST"])
def research_question():
    """Deep multi-step autonomous research."""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    agent = get_agent()

    # Collect reasoning steps for the response
    steps_log = []

    def on_step(step):
        steps_log.append({
            "step": step.step_number,
            "thought": step.thought[:500] if step.thought else "",
            "action": step.action,
            "action_input": step.action_input,
            "observation": step.observation[:500] if step.observation else "",
            "is_final": step.is_final,
        })

    result = agent.research(question, on_step=on_step)

    return jsonify({
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
        "reasoning_steps": steps_log,
        "steps_taken": result.steps_taken,
        "response_time": result.total_time,
        "mode": "research",
    })


# ── Documents & Stats ────────────────────────────────────────────────────────

@app.route("/api/documents", methods=["GET"])
def list_documents():
    agent = get_agent()
    docs = agent.get_document_list()
    return jsonify({"documents": docs, "count": len(docs)})


@app.route("/api/upload", methods=["POST"])
def upload_document():
    global AGENT
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    AGENT = None
    agent = get_agent()

    return jsonify({
        "message": f"Uploaded {filename} and re-indexed",
        "documents": agent.get_document_list(),
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    agent = get_agent()
    stats = agent.get_stats()
    return jsonify({
        "documents": len(agent.get_document_list()),
        "chunks": len(agent.base_agent.processor.documents),
        "vocabulary_size": (
            len(agent.base_agent.vector_store.vectorizer.vocabulary_)
            if agent.base_agent.vector_store._is_fitted
            else 0
        ),
        "mode": stats.get("mode", "strategic"),
        "max_reasoning_steps": stats.get("max_reasoning_steps", 10),
    })


if __name__ == "__main__":
    print("\nStarting Strategic Research Agent...")
    get_agent()
    app.run(host="0.0.0.0", port=5001, debug=False)
