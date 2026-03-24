"""
agent.py
Main orchestrator for the AI Leadership Insight Agent.

Ties together:
- Document ingestion and chunking
- TF-IDF vector indexing
- Query expansion
- Multi-query retrieval
- Answer generation

Usage:
    from agent import LeadershipInsightAgent
    
    agent = LeadershipInsightAgent(document_folder="./documents")
    response = agent.ask("What is our current revenue trend?")
    print(response.answer)
"""

import os
import time
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_expander import QueryExpander
from answer_generator import AnswerGenerator, AgentResponse


class LeadershipInsightAgent:
    """
    AI-powered assistant that answers leadership questions
    grounded in internal company documents.
    """

    def __init__(
        self,
        document_folder: str = "./documents",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        answer_mode: str = "local",
    ):
        self.document_folder = document_folder
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = VectorStore(
            max_features=8000,
            ngram_range=(1, 2),
        )
        self.query_expander = QueryExpander()
        self.answer_gen = AnswerGenerator(mode=answer_mode)
        self._is_ready = False

    def initialize(self) -> dict:
        """
        Ingest documents and build the search index.
        Returns stats about the ingestion.
        """
        print(f"\n{'='*60}")
        print("AI Leadership Insight Agent – Initializing")
        print(f"{'='*60}")
        print(f"Document folder: {self.document_folder}\n")

        t0 = time.time()

        print("Step 1: Ingesting documents...")
        chunks = self.processor.ingest_folder(self.document_folder)

        print("\nStep 2: Building search index...")
        self.vector_store.index_documents(chunks)

        elapsed = time.time() - t0
        self._is_ready = True

        stats = {
            "documents_processed": len(set(c.source_file for c in chunks)),
            "total_chunks": len(chunks),
            "vocabulary_size": len(self.vector_store.vectorizer.vocabulary_),
            "indexing_time_seconds": round(elapsed, 2),
        }

        print(f"\n{'='*60}")
        print(f"Ready! Processed {stats['documents_processed']} documents → "
              f"{stats['total_chunks']} chunks in {stats['indexing_time_seconds']}s")
        print(f"{'='*60}\n")

        return stats

    def ask(self, question: str, top_k: int = 10) -> AgentResponse:
        """
        Answer a leadership question using the indexed documents.

        Pipeline:
        1. Expand query into variations
        2. Multi-query retrieval with RRF fusion
        3. Expand context with neighboring chunks
        4. Generate grounded answer
        """
        if not self._is_ready:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        query_variations = self.query_expander.expand(question)

        if len(query_variations) > 1:
            results = self.vector_store.multi_query_search(
                queries=query_variations,
                top_k=top_k,
            )
        else:
            results = self.vector_store.search(
                query=question,
                top_k=top_k,
            )

        results = self.vector_store.expand_context(results, window=1)

        response = self.answer_gen.generate(
            question=question,
            results=results,
            query_variations=query_variations,
        )

        return response

    def get_document_list(self) -> list[str]:
        """Return list of indexed document filenames."""
        return sorted(set(c.source_file for c in self.processor.documents))


#  CLI interface

def main():
    """Interactive CLI for the Leadership Insight Agent."""
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "./documents"
    mode = os.environ.get("ANSWER_MODE", "local")

    agent = LeadershipInsightAgent(document_folder=folder, answer_mode=mode)
    stats = agent.initialize()

    print("\nIndexed documents:")
    for doc in agent.get_document_list():
        print(f"  • {doc}")

    print("\n" + "="*60)
    print("Ask leadership questions (type 'quit' to exit)")
    print("="*60 + "\n")

    while True:
        try:
            question = input("📊 Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        t0 = time.time()
        response = agent.ask(question)
        elapsed = time.time() - t0

        print(f"\n{'─'*60}")
        print(f"Confidence: {response.confidence.upper()}")
        print(f"Query variations used: {response.query_variations}")
        print(f"{'─'*60}")
        print(f"\n{response.answer}\n")
        print(f"{'─'*60}")
        print(f"Sources:")
        for s in response.sources:
            print(f"  • {s['file']} → {s['section']} (score: {s['relevance_score']})")
        print(f"Response time: {elapsed:.2f}s")
        print(f"{'─'*60}\n")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
