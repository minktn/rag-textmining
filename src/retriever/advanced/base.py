"""
Base Advanced Retriever Interface
=================================
Định nghĩa Abstract Base Class cho tất cả các phương thức truy xuất nâng cao (Advanced Retrieval).
Áp dụng Strategy Pattern & Open-Closed Principle (OCP):
Mỗi phương thức (như rag_fusion, self_rag, ...) sẽ tự đóng gói toàn bộ logic khởi tạo,
sinh câu hỏi/đánh giá, và truy xuất của mình mà không làm thay đổi Retriever pipeline chính.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAdvancedRetriever(ABC):
    """Abstract Base Class cho một chiến lược truy xuất nâng cao."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        retriever: Any,
        normalized_query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Thực thi chiến lược truy xuất nâng cao và trả về kết quả chuẩn dict:
        {
            "query": ...,
            "selected_chunks": ...,
            "expanded_chunks": ...,
            "context": ...,
            ...
        }
        """
        pass
