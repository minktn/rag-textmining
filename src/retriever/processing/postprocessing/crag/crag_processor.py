"""
CRAG Processor (Corrective Retrieval-Augmented Generation)
===========================================================
Pipeline xử lý CRAG hoàn chỉnh cho lĩnh vực pháp luật Việt Nam:
1. Đánh giá tài liệu nội bộ bằng BamiBERT ViLegalNLI (2 nhãn → Confidence score).
2. Kích hoạt hành động tương ứng:
   - Correct: Tinh lọc tài liệu nội bộ (Knowledge Refinement).
   - Incorrect: Bỏ qua tài liệu nội bộ, tra cứu cơ sở dữ liệu luật qua Vietnamese Law MCP.
   - Ambiguous: Kết hợp tài liệu nội bộ đã tinh lọc + tra cứu bổ sung qua Vietnamese Law MCP.
3. Trả về tập ngữ cảnh hoàn thiện kèm metadata chi tiết phục vụ sinh câu trả lời.
"""

import logging
from typing import Any, Dict, List, Optional

from src.retriever.web_search import VietnameseLawWebSearch
from .evaluator import BamiBERTRetrievalEvaluator
from .refiner import KnowledgeRefiner

logger = logging.getLogger(__name__)


class CRAGProcessor:
	"""Bộ điều phối Corrective RAG (CRAG) cho bài toán RAG Pháp luật Việt Nam."""

	def __init__(
		self,
		evaluator: Optional[BamiBERTRetrievalEvaluator] = None,
		web_search: Optional[VietnameseLawWebSearch] = None,
		refiner: Optional[KnowledgeRefiner] = None,
		upper_threshold: float = 0.75,
		lower_threshold: float = 0.35,
		mcp_top_k: int = 3,
	):
		self.evaluator = evaluator or BamiBERTRetrievalEvaluator(
			upper_threshold=upper_threshold,
			lower_threshold=lower_threshold,
		)
		self.refiner = refiner or KnowledgeRefiner(evaluator=self.evaluator)
		self.web_search = web_search or VietnameseLawWebSearch()
		self.upper_threshold = upper_threshold
		self.lower_threshold = lower_threshold
		self.mcp_top_k = mcp_top_k

	def process(
		self,
		query: str,
		retrieved_chunks: List[Dict[str, Any]],
		upper_threshold: Optional[float] = None,
		lower_threshold: Optional[float] = None,
	) -> Dict[str, Any]:
		"""Thực thi pipeline CRAG: Evaluate → Trigger Action → Refine/Search → Combine.

		Parameters
		----------
		query : str
			Câu hỏi của người dùng.
		retrieved_chunks : List[Dict[str, Any]]
			Danh sách các chunks ban đầu truy xuất từ database nội bộ (Qdrant).
		upper_threshold : Optional[float]
			Ngưỡng trên cho trạng thái Correct.
		lower_threshold : Optional[float]
			Ngưỡng dưới cho trạng thái Incorrect.

		Returns
		-------
		Dict[str, Any]
			{
				'query': str,
				'action': 'correct' | 'ambiguous' | 'incorrect',
				'processed_chunks': List[Dict[str, Any]],
				'internal_chunks': List[Dict[str, Any]],
				'external_chunks': List[Dict[str, Any]],
				'evaluation_details': List[Dict[str, Any]],
			}
		"""
		u_th = upper_threshold if upper_threshold is not None else self.upper_threshold
		l_th = lower_threshold if lower_threshold is not None else self.lower_threshold

		# Bước 1: Đánh giá tài liệu nội bộ bằng Evaluator (BamiBERT-ViLegalNLI)
		overall_action, evaluated_chunks = self.evaluator.evaluate_chunks(
			query=query,
			chunks=retrieved_chunks,
			upper_th=u_th,
			lower_th=l_th,
		)

		internal_refined_chunks = []
		external_chunks = []
		final_chunks = []

		logger.info(f"[CRAG] Query: '{query}' | Overall Action: {overall_action.upper()}")

		# Bước 2: Phân nhánh hành động (Triggering Actions)
		if overall_action == "correct":
			# Chỉ giữ các chunks đạt "correct" hoặc có điểm cao, sau đó tinh lọc câu
			valid_internal = [c for c in evaluated_chunks if c.get("crag_action") != "incorrect"]
			if not valid_internal:
				valid_internal = evaluated_chunks
			internal_refined_chunks = self.refiner.refine_chunks(query, valid_internal, min_score_threshold=l_th)
			final_chunks = internal_refined_chunks

		elif overall_action == "incorrect":
			# Tài liệu nội bộ sai lệch -> Kích hoạt Web Search / MCP
			logger.info("[CRAG] Kích hoạt Vietnamese Law MCP Search do tài liệu nội bộ không liên quan...")
			mcp_res = self.web_search.retrieve(query, top_k=self.mcp_top_k)
			external_chunks = mcp_res.get("context_chunks", [])
			final_chunks = external_chunks

		elif overall_action == "ambiguous":
			# Tài liệu nội bộ phân vân -> Kết hợp tài liệu nội bộ đã lọc + MCP Web Search
			logger.info("[CRAG] Kết hợp Internal Knowledge + Vietnamese Law MCP Search...")
			valid_internal = [c for c in evaluated_chunks if c.get("crag_action") != "incorrect"]
			if not valid_internal:
				valid_internal = evaluated_chunks
			internal_refined_chunks = self.refiner.refine_chunks(query, valid_internal, min_score_threshold=l_th)

			mcp_res = self.web_search.retrieve(query, top_k=self.mcp_top_k)
			external_chunks = mcp_res.get("context_chunks", [])

			# Kết hợp cả hai nguồn tri thức
			final_chunks = internal_refined_chunks + external_chunks

		return {
			"query": query,
			"action": overall_action,
			"processed_chunks": final_chunks,
			"internal_chunks": internal_refined_chunks,
			"external_chunks": external_chunks,
			"evaluation_details": evaluated_chunks,
		}
