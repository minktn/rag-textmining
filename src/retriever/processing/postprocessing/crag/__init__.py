"""
CRAG Module (Corrective Retrieval-Augmented Generation)
======================================================
Pipeline xử lý CRAG cho bài toán RAG Pháp luật Việt Nam:
- BamiBERTRetrievalEvaluator: Đánh giá độ tin cậy của tài liệu qua FastAPI NLI Server (port 8001).
- verify_nli_service: Kiểm tra trạng thái NLI server trước khi khởi chạy.
- KnowledgeRefiner: Phân rã và tinh lọc dải tri thức (Knowledge Strips).
- CRAGProcessor: Bộ điều phối trung tâm kết nối Qdrant Retrieval, NLI Evaluator và Vietnamese Law MCP.
- utils: Các hàm tiện ích tách câu và lọc dải tri thức liên quan.
"""

from .evaluator import BamiBERTRetrievalEvaluator, verify_nli_service
from .refiner import KnowledgeRefiner
from .crag_processor import CRAGProcessor
from .utils import extract_strips_from_passage, select_relevants

__all__ = [
	"BamiBERTRetrievalEvaluator",
	"verify_nli_service",
	"KnowledgeRefiner",
	"CRAGProcessor",
	"extract_strips_from_passage",
	"select_relevants",
]
