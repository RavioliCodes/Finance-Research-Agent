"""
demo.py
Demonstrates the AI Leadership Insight Agent with sample questions.
Run: python demo.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from agent import LeadershipInsightAgent


def run_demo():
    print("=" * 70)
    print("   AI LEADERSHIP INSIGHT AGENT – DEMO")
    print("=" * 70)

    # Initialize
    agent = LeadershipInsightAgent(
        document_folder=os.path.join(os.path.dirname(__file__), "documents"),
        answer_mode="local",
    )
    stats = agent.initialize()

    print(f"\nDocuments loaded: {agent.get_document_list()}\n")

    # Demo questions
    questions = [
        "What is our current revenue trend?",
        "Which departments are underperforming?",
        "What were the key risks highlighted in the last quarter?",
        "What is our strategy for international expansion?",
        "What is the company's cash position?",
        "How is our customer success team performing?",
        "What are our financial targets for the next 3 years?",
        "What M&A targets are being considered?",
    ]

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n{'━' * 70}")
        print(f"  Q{i}: {q}")
        print(f"{'━' * 70}")

        response = agent.ask(q)

        print(f"\n  Confidence: {response.confidence.upper()}")
        print(f"  Queries used: {response.query_variations[:3]}")
        print(f"\n{response.answer}")
        print(f"\n  Sources:")
        for s in response.sources:
            print(f"    • {s['file']} → {s['section']} (score: {s['relevance_score']})")

        results.append({
            "question": q,
            "confidence": response.confidence,
            "sources_count": len(response.sources),
            "answer_length": len(response.answer),
        })

    # Summary
    print(f"\n\n{'=' * 70}")
    print("   DEMO SUMMARY")
    print(f"{'=' * 70}")
    high = sum(1 for r in results if r["confidence"] == "high")
    med = sum(1 for r in results if r["confidence"] == "medium")
    low = sum(1 for r in results if r["confidence"] == "low")
    print(f"  Questions answered: {len(results)}")
    print(f"  Confidence: {high} high / {med} medium / {low} low")
    print(f"  Average answer length: {sum(r['answer_length'] for r in results) // len(results)} chars")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_demo()
