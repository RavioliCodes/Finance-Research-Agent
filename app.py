"""
app.py
Flask web application for the AI Leadership Insight Agent.
Provides a REST API and serves the web UI.
"""

import os
import json
import time
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename

from agent import LeadershipInsightAgent

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
        mode = os.environ.get("ANSWER_MODE", "local")
        AGENT = LeadershipInsightAgent(document_folder=doc_folder, answer_mode=mode)
        AGENT.initialize()
    return AGENT


@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """Answer a leadership question."""
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
    })


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """List indexed documents."""
    agent = get_agent()
    docs = agent.get_document_list()
    return jsonify({"documents": docs, "count": len(docs)})


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """Upload a new document and re-index."""
    global AGENT
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    # Force re-initialization
    AGENT = None
    agent = get_agent()

    return jsonify({
        "message": f"Uploaded {filename} and re-indexed",
        "documents": agent.get_document_list(),
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get agent statistics."""
    agent = get_agent()
    return jsonify({
        "documents": len(agent.get_document_list()),
        "chunks": len(agent.processor.documents),
        "vocabulary_size": len(agent.vector_store.vectorizer.vocabulary_) if agent.vector_store._is_fitted else 0,
        "mode": agent.answer_gen.mode,
    })


if __name__ == "__main__":
    print("\n🚀 Starting AI Leadership Insight Agent...")
    get_agent()  # Pre-initialize
    app.run(host="0.0.0.0", port=5000, debug=False)
