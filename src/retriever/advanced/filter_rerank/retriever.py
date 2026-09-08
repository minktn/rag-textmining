"""
Filter-then-Rerank Advanced Retriever Strategy
==============================================
Encapsulates the Filter-then-Rerank framework (EMNLP 2023) as a first-class
Advanced Retrieval strategy conforming to BaseAdvancedRetriever.

Architecture:
1. Candidate Search: High-recall dense vector retrieval (e.g. 20 chunks).
2. SLM Filter (4-bit): High-throughput preliminary candidate filtering via local 4-bit SLM logits.
3. LLM Reranker: Multi-choice reasoning for hard/ambiguous samples via Cloud LLM.
4. Reference Expansion: Automatically enriches context with referenced legal articles.

Bypasses the standard pipeline's BAAI/bge-reranker-v2-m3 completely, saving ~2.2GB VRAM.
"""

import logging
from typing import Any, Dict, List, Optional

from src.retriever.advanced.base import BaseAdvancedRetriever
from src.retriever.advanced.factory import register_advanced
from .filter import SLMFilter
from .processor import FilterReranker
from .reranker import LLMReranker

logger = logging.getLogger(__name__)


@register_advanced("filter_rerank")
class FilterRerankRetriever(BaseAdvancedRetriever):
    """Chiến lược truy xuất nâng cao Filter-then-Rerank (SLM Filter 4-bit + Cloud LLM Reranker)."""

    def __init__(
        self,
        sub_llm_manager: Optional[Any] = None,
        tau_high: float = 0.70,
        tau_low: float = 0.20,
        **kwargs,
    ):
        self.sub_llm_manager = sub_llm_manager
        self.slm_filter = SLMFilter(tau_high=tau_high, tau_low=tau_low)
        self.llm_reranker = LLMReranker(sub_llm_manager=sub_llm_manager)
        self.processor = FilterReranker(
            slm_filter=self.slm_filter,
            llm_reranker=self.llm_reranker,
            sub_llm_manager=sub_llm_manager,
            tau_high=tau_high,
            tau_low=tau_low,
        )
        logger.info("[FilterRerankRetriever] Advanced strategy initialized (4-bit SLM + LLM Reranker).")

    def retrieve(
        self,
        query: str,
        retriever: Any,
        normalized_query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Thực thi chiến lược truy xuất Filter-then-Rerank hoàn chỉnh:
        Dense Search (Recall cao) -> SLM Filter 4-bit -> LLM Reranker -> Reference Expansion.
        """
        norm_q = normalized_query or retriever.normalize_query(query)
        f = filters if filters is not None else retriever.extract_filters(norm_q)
        query_filter = retriever.build_query_filter(f)
        filter_relaxed = False

        # 1. Truy vấn Dense candidates (Ưu tiên Local > Cloud)
        query_vector = retriever.embedder.embed_single(norm_q)
        candidates = retriever._query_dense_candidates(
            query_vector=query_vector,
            query_filter=query_filter,
        )

        if not candidates and query_filter is not None and retriever.relax_filter_on_empty:
            filter_relaxed = True
            candidates = retriever._query_dense_candidates(
                query_vector=query_vector,
                query_filter=None,
            )

        dense_chunks = [
            retriever._format_chunk(candidate, source="dense")
            for candidate in candidates
        ]

        logger.info(
            f"[FilterRerankRetriever] Retrieved {len(dense_chunks)} dense candidates. "
            f"Routing directly to 2-stage Filter-then-Rerank (bge-reranker-v2-m3 bypassed)."
        )

        # 2. Thực thi Filter-then-Rerank 2 tầng (bypasses bge-reranker-v2-m3)
        selected_chunks = self.processor.process(
            query=query,
            chunks=dense_chunks,
            retriever=retriever,
            top_k=retriever.rerank_limit,
        )

        # 3. Mở rộng điều luật tham chiếu
        expanded_chunks = retriever.expand_references(selected_chunks)

        return retriever._build_result(
            query=query,
            normalized_query=norm_q,
            filters=f,
            filter_applied=query_filter is not None and not filter_relaxed,
            filter_relaxed=filter_relaxed,
            selected_chunks=selected_chunks,
            expanded_chunks=expanded_chunks,
        )
