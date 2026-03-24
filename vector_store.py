"""
vector_store.py
Lightweight vector store using TF-IDF embeddings + cosine similarity.
No external API keys required – runs entirely locally.

For production, this can be swapped with OpenAI/Cohere embeddings + FAISS/Chroma.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from document_processor import DocumentChunk


@dataclass
class SearchResult:
    """A single retrieval result with score."""
    chunk: DocumentChunk
    score: float

    def __repr__(self):
        return f"Result(score={self.score:.3f}, src={self.chunk.source_file}, section={self.chunk.section_header})"


class VectorStore:
    """
    TF-IDF based vector store for document retrieval.
    
    Uses scikit-learn's TfidfVectorizer with tuned parameters for
    enterprise document retrieval. Supports:
    - Semantic-ish search via TF-IDF + cosine similarity
    - Keyword boosting for financial/business terms
    - Multi-query retrieval with score fusion
    """

    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,         # apply log normalization to tf
            min_df=1,
            max_df=0.95,
            dtype=np.float32,
        )
        self.chunks: list[DocumentChunk] = []
        self.tfidf_matrix = None
        self._is_fitted = False

    def index_documents(self, chunks: list[DocumentChunk]) -> None:
        """Build the TF-IDF index from document chunks."""
        self.chunks = chunks
        texts = [self._prepare_text(c) for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        print(f"Indexed {len(chunks)} chunks | Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.05,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most relevant chunks for a query.
        """
        if not self._is_fitted:
            raise RuntimeError("Index not built. Call index_documents() first.")

        query_vec = self.vectorizer.transform([self._prepare_query(query)])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # rank by score
        ranked_indices = np.argsort(scores)[::-1]

        results: list[SearchResult] = []
        seen_texts = set()

        for idx in ranked_indices:
            if scores[idx] < score_threshold:
                break
            # deduplicate near-identical chunks
            text_sig = self.chunks[idx].text[:200]
            if text_sig in seen_texts:
                continue
            seen_texts.add(text_sig)

            results.append(SearchResult(chunk=self.chunks[idx], score=float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 5,
        score_threshold: float = 0.05,
    ) -> list[SearchResult]:
        """
        Search with multiple query variations and fuse results via
        Reciprocal Rank Fusion (RRF).
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}
        k_rrf = 60  # RRF constant

        for q in queries:
            results = self.search(q, top_k=top_k * 2, score_threshold=score_threshold)
            for rank, r in enumerate(results):
                cid = r.chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k_rrf + rank + 1)
                if cid not in chunk_map or r.score > chunk_map[cid].score:
                    chunk_map[cid] = r

        # sort by fused score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            SearchResult(chunk=chunk_map[cid].chunk, score=rrf_scores[cid])
            for cid in sorted_ids
        ]

    def expand_context(
        self,
        results: list[SearchResult],
        window: int = 1,
    ) -> list[SearchResult]:
        """Expand each result by including `window` neighboring chunks
        from the same source file.  This provides the LLM with surrounding
        context so it can see the full picture rather than isolated fragments.

        Neighbors are inserted directly before/after the matched chunk in
        the returned list.  Duplicates are suppressed.
        """
        if not self.chunks:
            return results

        # Build a quick lookup: (source_file, chunk_index) → position in self.chunks
        idx_map: dict[tuple[str, int], int] = {}
        for pos, c in enumerate(self.chunks):
            idx_map[(c.source_file, c.chunk_index)] = pos

        seen_ids: set[str] = set()
        expanded: list[SearchResult] = []

        for r in results:
            key = (r.chunk.source_file, r.chunk.chunk_index)
            pos = idx_map.get(key)
            if pos is None:
                if r.chunk.chunk_id not in seen_ids:
                    expanded.append(r)
                    seen_ids.add(r.chunk.chunk_id)
                continue

            # Gather the chunk itself plus neighbors.
            neighbors = range(max(0, pos - window), min(len(self.chunks), pos + window + 1))
            for n_pos in neighbors:
                nc = self.chunks[n_pos]
                # Only include neighbors from the same source file.
                if nc.source_file != r.chunk.source_file:
                    continue
                if nc.chunk_id in seen_ids:
                    continue
                seen_ids.add(nc.chunk_id)
                # Neighbors get a slightly lower score to preserve ranking.
                score = r.score if n_pos == pos else r.score * 0.5
                expanded.append(SearchResult(chunk=nc, score=score))

        return expanded

    @staticmethod
    def _prepare_text(chunk: DocumentChunk) -> str:
        """Prepare chunk text for indexing (include section header for context)."""
        parts = []
        if chunk.section_header:
            parts.append(chunk.section_header)
        parts.append(chunk.text)
        return " ".join(parts)

    @staticmethod
    def _prepare_query(query: str) -> str:
        """Prepare query for search."""
        return query.strip()
