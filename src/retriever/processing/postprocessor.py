"""
Postprocessor
=============
Lớp chung duy nhất điều phối toàn bộ các phương thức hậu xử lý chunks (Postprocessing).
Gọi đến các module phương thức đã được đóng gói độc lập trong src/retriever/processing/postprocessing/.
"""

import logging
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class Postprocessor:
    """Điều phối và thực thi các phương thức hậu xử lý chunks (ví dụ: filter_rerank, crag, prompt_compression)."""

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        retriever: Optional[Any] = None,
    ):
        self.methods = [m.lower().strip() for m in (methods or [])]
        self.retriever = retriever

    def process(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        retriever: Optional[Any] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Thực thi tuần tự các bước hậu xử lý đã được chỉ định."""
        if not self.methods or not chunks:
            return chunks

        active_retriever = retriever or self.retriever
        current_chunks = chunks

        for method in self.methods:
            if method == "base":
                continue

            logger.info(f"[Postprocessor] Thực thi phương thức: '{method}'")
            try:
                current_chunks = self._dispatch(method, query, current_chunks, active_retriever, **kwargs)
            except Exception as e:
                logger.error(f"[Postprocessor] Hậu xử lý '{method}' thất bại: {e}. Giữ nguyên chunks.")

        return current_chunks

    def _dispatch(
        self,
        method: str,
        query: str,
        chunks: List[Dict[str, Any]],
        retriever: Optional[Any] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Uỷ quyền xử lý cho module đóng gói tương ứng."""
        if method == "filter_rerank":
            from .postprocessing.filter_rerank import FilterReranker

            sub_llm = getattr(retriever, "sub_llm_manager", None) if retriever else None
            top_k = getattr(retriever, "rerank_limit", None) if retriever else None
            processor = FilterReranker(sub_llm_manager=sub_llm)
            return processor.process(query, chunks, top_k=top_k)

        elif method == "crag":
            from .postprocessing.crag import CRAGProcessor

            processor = CRAGProcessor()
            result = processor.process(query, chunks)
            logger.info(f"[Postprocessor] CRAG hoàn tất với action: {result.get('action')}")
            return result.get("processed_chunks", chunks)

        elif method == "prompt_compression":
            from .postprocessing.prompt_compression.compressor import LongLLMLinguaCompressor

            compressor = LongLLMLinguaCompressor(device=settings.DEVICE)
            contexts = [c.get("content", "") for c in chunks]
            result = compressor.compress_retrieved_context(query, contexts)
            logger.info(
                f"[Postprocessor] Prompt compression: {result['origin_tokens']} -> "
                f"{result['compressed_tokens']} tokens (tiết kiệm {result['saving_ratio']:.1%})"
            )
            for c in chunks:
                c["_compressed"] = True
            if chunks:
                chunks[0]["_compressed_context"] = result["compressed_context"]
            return chunks

        logger.warning(f"[Postprocessor] Phương thức hậu xử lý '{method}' chưa được triển khai. Bỏ qua.")
        return chunks
