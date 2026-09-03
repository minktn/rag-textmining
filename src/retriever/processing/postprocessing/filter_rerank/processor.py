"""
Filter-then-Rerank Processor
============================
Bộ điều phối toàn diện cho mô hình Filter-then-Rerank (EMNLP 2023):
Kết hợp sức mạnh phân loại nhanh của Small Language Model (SLM) local
với năng lực suy luận chuyên sâu của Large Language Model (NVIDIA) cho các mẫu khó.
"""

import logging
from typing import Any, Dict, List, Optional

from .filter import SLMFilter
from .reranker import LLMReranker

logger = logging.getLogger(__name__)


class FilterReranker:
    """Bộ điều phối quy trình Filter-then-Rerank cho bước hậu xử lý RAG."""

    def __init__(
        self,
        slm_filter: Optional[SLMFilter] = None,
        llm_reranker: Optional[LLMReranker] = None,
        sub_llm_manager: Optional[Any] = None,
        tau_high: float = 0.70,
        tau_low: float = 0.30,
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
        Thực thi quy trình Filter-then-Rerank trên tập ứng viên chunks.

        Parameters
        ----------
        query : str
            Câu hỏi của người dùng.
        chunks : List[Dict[str, Any]]
            Danh sách chunks thu được từ bước truy xuất dense / rerank ban đầu.
        top_k : Optional[int]
            Số lượng chunks tối đa muốn giữ lại.

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách chunks đã qua lọc và tái xếp hạng.
        """
        if not chunks:
            return []

        # ── Giai đoạn 1: Lọc bằng SLM Local ────────────────────────
        easy_chunks, hard_chunks, dropped_chunks = self.slm_filter.filter_chunks(query, chunks)

        for c in easy_chunks:
            c["final_score"] = c.get("slm_score", 1.0)
            c["source"] = "slm_filter_easy_kept"

        # ── Giai đoạn 2: Tái thẩm định Hard Samples bằng LLM Cloud (NVIDIA) ─
        if hard_chunks:
            reranked_hard = self.llm_reranker.rerank_hard_samples(query, hard_chunks)
        else:
            reranked_hard = []

        # ── Giai đoạn 3: Hợp nhất & Sắp xếp ────────────────────────
        combined = easy_chunks + reranked_hard

        # Fallback: nếu toàn bộ bị drop do threshold quá gắt, giữ lại các chunks từ dropped có điểm cao nhất
        if not combined and dropped_chunks:
            logger.warning("[FilterReranker] Toàn bộ chunks bị loại bỏ bởi ngưỡng. Kích hoạt fallback giữ lại ứng viên tốt nhất.")
            for c in dropped_chunks:
                c["final_score"] = c.get("slm_score", 0.0)
                c["source"] = "slm_filter_fallback"
            dropped_chunks.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            combined = dropped_chunks[:3]

        # Sắp xếp giảm dần theo điểm số cuối cùng
        combined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        effective_top_k = top_k or (getattr(retriever, "rerank_limit", None) if retriever else None)
        if effective_top_k and len(combined) > effective_top_k:
            combined = combined[:effective_top_k]

        logger.info(
            f"[FilterReranker] Hoàn tất: Giữ lại {len(combined)} chunks "
            f"(Easy: {len(easy_chunks)}, Hard-reranked: {len(reranked_hard)}, Dropped: {len(dropped_chunks)})"
        )
        return combined
