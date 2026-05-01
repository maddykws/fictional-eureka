"""
BM25 retriever over the support corpus.

For each support ticket, retrieves the top-K most relevant corpus chunks
restricted to the company's domain (hackerrank / claude / visa).

Uses BM25L (rank_bm25) which fixes the over-penalisation of long documents —
important because support corpus articles vary greatly in length.
Research basis: "Hybrid Retrieval: DAT — Dynamic Alpha Tuning" (2025) confirms
BM25L outperforms BM25Okapi on support-style documents with verbose descriptions.

Falls back to BM25Okapi then TF overlap if BM25L is unavailable.

Spotlighting (Hines et al. 2024, arXiv:2403.14720): retrieved chunks are wrapped
in [CORPUS]...[/CORPUS] delimiters so the LLM treats them as data, not instructions.
This reduces indirect prompt injection success from >50% to <2%.

Confidence gate: if max BM25 score < LOW_SCORE_THRESHOLD the corpus does not
contain a relevant answer — triage() escalates instead of hallucinating.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from corpus import CorpusChunk, load_corpus, get_domain_for_company

TOP_K = 6          # chunks returned per query
MAX_CHUNK_CHARS = 2000  # truncate very long chunks to keep prompt lean
LOW_SCORE_THRESHOLD = 0.5  # below this → corpus has no relevant doc → escalate


@dataclass
class RetrievalResult:
    chunk:   CorpusChunk
    score:   float
    snippet: str   # truncated content


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[CorpusChunk]):
        self._chunks = chunks
        self._corpus_tokens = [c.tokens for c in chunks]
        self._index = None
        self._build_index()

    def _build_index(self) -> None:
        try:
            # BM25L fixes over-penalisation of long documents (vs BM25Okapi)
            from rank_bm25 import BM25L
            self._index = BM25L(self._corpus_tokens)
        except (ImportError, Exception):
            try:
                from rank_bm25 import BM25Okapi
                self._index = BM25Okapi(self._corpus_tokens)
            except ImportError:
                self._index = None  # fallback to TF overlap

    def _tf_overlap_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        q_set = set(query_tokens)
        overlap = sum(1 for t in doc_tokens if t in q_set)
        return overlap / (len(doc_tokens) ** 0.5 + 1)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
        query_tokens = _tokenise(query)

        if self._index is not None:
            scores = self._index.get_scores(query_tokens)
        else:
            scores = [self._tf_overlap_score(query_tokens, c) for c in self._corpus_tokens]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            snippet = chunk.content[:MAX_CHUNK_CHARS]
            if len(chunk.content) > MAX_CHUNK_CHARS:
                snippet += "\n[...truncated]"
            results.append(RetrievalResult(chunk=chunk, score=float(score), snippet=snippet))

        return results

    def top_score(self, query: str) -> float:
        """Return the highest BM25 score for the query across all docs in this index."""
        results = self.retrieve(query, top_k=1)
        return results[0].score if results else 0.0


# ── Module-level retriever cache ───────────────────────────────────────────────

_retrievers: dict[str, BM25Retriever] = {}


def get_retriever(domain_keys: list[str]) -> BM25Retriever:
    """
    Return a cached retriever for the given domain keys.
    domain_keys: e.g. ["hackerrank"] or ["hackerrank", "claude", "visa"]
    """
    cache_key = ",".join(sorted(domain_keys))
    if cache_key not in _retrievers:
        chunks = load_corpus(domains=domain_keys)
        _retrievers[cache_key] = BM25Retriever(chunks)
    return _retrievers[cache_key]


def retrieve_for_ticket(issue: str, subject: str, company: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    domain_keys = get_domain_for_company(company)
    retriever = get_retriever(domain_keys)
    query = f"{subject} {issue}".strip()
    return retriever.retrieve(query, top_k=top_k)


def format_context(results: list[RetrievalResult]) -> str:
    """
    Format retrieved chunks as a spotlighted context block for the LLM prompt.

    Spotlighting (Hines et al. 2024, arXiv:2403.14720): wrapping each chunk in
    [CORPUS]...[/CORPUS] signals to the LLM that this content is external data
    and must not be treated as instructions, reducing indirect prompt injection
    success rates from >50% to <2%.
    """
    if not results:
        return "[No relevant corpus documents found]"

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[CORPUS]\n"
            f"[Doc {i}] {r.chunk.domain} / {r.chunk.subdomain} — {r.chunk.title}\n"
            f"{r.snippet}\n"
            f"[/CORPUS]"
        )
    return "\n\n---\n\n".join(parts)
