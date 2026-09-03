"""
RAG-Fusion Advanced Retriever Strategy
======================================
Đóng gói hoàn chỉnh quy trình truy xuất RAG-Fusion (Multi-query + Reciprocal Rank Fusion)
tuân thủ giao diện BaseAdvancedRetriever.
"""

import logging
from typing import Any, Dict, List, Optional

from src.retriever.advanced.base import BaseAdvancedRetriever
from .query_generator import RAGFusionQueryGenerator
from .rag_fusion import RAGFusionProcessor

logger = logging.getLogger(__name__)


class RAGFusionRetriever(BaseAdvancedRetriever):
    """Chiến lược truy xuất nâng cao sử dụng RAG-Fusion."""

    def __init__(
        self,
        sub_llm_manager: Optional[Any] = None,
        k: int = 60,
        num_queries: int = 4,
        original_query_weight: float = 3.0,
        **kwargs,
    ):
        self.sub_llm_manager = sub_llm_manager
        self.k = k
        self.num_queries = num_queries
        self.original_query_weight = original_query_weight

        q_gen = (
            RAGFusionQueryGenerator(llm_manager=self.sub_llm_manager)
            if self.sub_llm_manager is not None
            else None
        )
        self.processor = RAGFusionProcessor(query_generator=q_gen, k=self.k)

    def retrieve(
        self,
        query: str,
        retriever: Any,
        normalized_query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Thực thi truy xuất RAG-Fusion đa truy vấn độc lập và dung hợp kết quả."""
        norm_q = normalized_query or retriever.normalize_query(query)
        f = filters if filters is not None else retriever.extract_filters(norm_q)
        query_filter = retriever.build_query_filter(f)

        def _core_retrieve_fn(q: str) -> List[Dict[str, Any]]:
            """Hàm truy xuất dense ứng viên cho từng sub-query."""
            q_norm = retriever.normalize_query(q)
            q_vec = retriever.embedder.embed_single(q_norm)
            return retriever._query_dense_candidates(
                query_vector=q_vec,
                query_filter=query_filter,
            )

        all_queries, fused_docs = self.processor.process(
            query=norm_q,
            retrieve_fn=_core_retrieve_fn,
            num_queries=self.num_queries,
            original_query_weight=self.original_query_weight,
            top_k=retriever.rerank_limit,
        )

        selected_chunks = [
            retriever._format_chunk(doc, source="rag_fusion") for doc in fused_docs
        ]
        expanded_chunks = retriever.expand_references(selected_chunks)

        return retriever._build_result(
            query=query,
            normalized_query=norm_q,
            filters=f,
            filter_applied=query_filter is not None,
            filter_relaxed=False,
            selected_chunks=selected_chunks,
            expanded_chunks=expanded_chunks,
        )
