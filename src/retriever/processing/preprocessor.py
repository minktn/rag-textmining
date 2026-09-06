"""
Preprocessor
============
Lớp chung duy nhất điều phối toàn bộ các phương thức tiền xử lý truy vấn (Preprocessing).
Gọi đến các module phương thức đã được đóng gói độc lập trong src/retriever/processing/preprocessing/.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class Preprocessor:
    """Điều phối và thực thi các phương thức tiền xử lý truy vấn (ví dụ: hyde, ...)."""

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
        retriever: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """Thực thi tuần tự các bước tiền xử lý đã được chỉ định."""
        if not self.methods:
            return query

        active_retriever = retriever or self.retriever
        processed_query = query

        for method in self.methods:
            logger.info(f"[Preprocessor] Thực thi phương thức: '{method}'")
            try:
                processed_query = self._dispatch(method, processed_query, active_retriever, **kwargs)
            except Exception as e:
                logger.error(f"[Preprocessor] Tiền xử lý '{method}' thất bại: {e}. Sử dụng query hiện tại.")

        return processed_query

    def _dispatch(
        self,
        method: str,
        query: str,
        retriever: Optional[Any] = None,
        **kwargs,
    ) -> str:
        """Uỷ quyền xử lý cho module đóng gói tương ứng."""
        if method == "hyde":
            from .preprocessing.hyde import HyDE

            embedder = getattr(retriever, "embedder", None) if retriever else None
            sub_llm = getattr(retriever, "sub_llm_manager", None) if retriever else None
            hyde = HyDE(llm_manager=sub_llm, embedder=embedder)
            result = hyde.process(query)

            if retriever is not None:
                retriever._hyde_embedding = result.get("embeddings")
            logger.info(
                f"[Preprocessor] HyDE hoàn tất: "
                f"{len(result.get('hypothetic_document', ''))} ký tự giả định."
            )
            return query

        logger.warning(f"[Preprocessor] Phương thức tiền xử lý '{method}' chưa được triển khai. Bỏ qua.")
        return query
