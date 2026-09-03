"""
Advanced Retrieval Architecture
===============================
Hệ thống các chiến lược truy xuất nâng cao thay thế pipeline chuẩn.
Áp dụng Strategy Pattern & Factory Pattern:
- BaseAdvancedRetriever: Interface chuẩn cho mọi phương thức advanced.
- get_advanced_retriever: Dynamic Factory khởi tạo strategy theo từ khoá.
- register_advanced: Decorator đăng ký phương thức mới.
"""

from .base import BaseAdvancedRetriever
from .factory import get_advanced_retriever, register_advanced

# Tự động đăng ký các module có sẵn
from .rag_fusion.retriever import RAGFusionRetriever
register_advanced("rag_fusion")(RAGFusionRetriever)

__all__ = [
	"BaseAdvancedRetriever",
	"get_advanced_retriever",
	"register_advanced",
	"RAGFusionRetriever",
]
