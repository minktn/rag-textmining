"""
Filter-then-Rerank Pipeline Coordinator
========================================
Orchestrates the two-stage Filter-then-Rerank framework (EMNLP 2023):
1. Stage 1 (Filter): Fast local SLM classifies candidates by confidence score.
2. Stage 2 (Rerank): Cloud LLM conducts in-depth multi-choice reasoning for hard samples.
3. Stage 3 (Merge): Assembles, scores, and sorts final ranked context.
"""

import logging
from typing import Any, Dict, List, Optional

from .filter import SLMFilter
from .reranker import LLMReranker

logger = logging.getLogger(__name__)


class FilterReranker:
    """Orchestrator for two-stage Filter-then-Rerank postprocessing."""

    def __init__(
        self,
        slm_filter: Optional[SLMFilter] = None,
        llm_reranker: Optional[LLMReranker] = None,
        sub_llm_manager: Optional[Any] = None,
        tau_high: float = 0.70,
        tau_low: float = 0.20,
        **kwargs,
    ):
        self.slm_filter = slm_filter or SLMFilter(tau_high=tau_high, tau_low=tau_low)
        self.llm_reranker = llm_reranker or LLMReranker(sub_llm_manager=sub_llm_manager)

    def process(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        retriever: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute the Filter-then-Rerank pipeline over retrieved candidates.

        Parameters
        ----------
        query : str
            User query.
        chunks : List[Dict[str, Any]]
            Candidate document chunks from dense/hybrid retrieval.
        retriever : Optional[Any]
            Retriever instance for configuration fallback.
        top_k : Optional[int]
            Maximum number of final chunks to return.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked, filtered candidates ordered by final relevance score.
        """
        if not chunks:
            return []

        # Stage 1: Preliminary screening via local SLM
        easy_chunks, hard_chunks, dropped_chunks = self.slm_filter.filter_chunks(query, chunks)

        for c in easy_chunks:
            c["final_score"] = c.get("slm_score", 1.0)
            c["source"] = "slm_filter_easy_kept"

        # Stage 2: In-depth multi-choice evaluation of hard samples via Cloud LLM
        reranked_hard = self.llm_reranker.rerank_hard_samples(query, hard_chunks) if hard_chunks else []

        # Stage 3: Merge, rank, and apply limit
        combined = easy_chunks + reranked_hard

        # Safety net: salvage top candidates if all chunks were filtered out
        if not combined and dropped_chunks:
            logger.warning("[FilterReranker] All candidates filtered by threshold. Reranking top-scoring fallbacks.")
            dropped_chunks.sort(key=lambda x: x.get("slm_score", 0.0), reverse=True)
            salvaged = dropped_chunks[:getattr(self.llm_reranker, "topk", 4)]
            reranked_hard = self.llm_reranker.rerank_hard_samples(query, salvaged)
            combined = reranked_hard or dropped_chunks[:3]

        combined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        effective_top_k = top_k or (getattr(retriever, "rerank_limit", None) if retriever else None)
        if effective_top_k and len(combined) > effective_top_k:
            combined = combined[:effective_top_k]

        logger.info(
            f"[FilterReranker] Finished: Retained {len(combined)} chunks "
            f"(Easy: {len(easy_chunks)}, Hard-reranked: {len(reranked_hard)}, Dropped: {len(dropped_chunks)})"
        )
        return combined
