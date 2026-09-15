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

	@staticmethod
	def _get_least_confident_negative_chunks(
		chunks: List[Dict[str, Any]], top_n: int = 3
	) -> List[Dict[str, Any]]:
		"""Trích xuất top N chunks nội bộ có kết quả phủ định với confidence score thấp nhất
		(tức NLI ít chắc chắn về việc phủ định nhất, có xác suất liên quan tiềm năng cao nhất)
		để phục vụ fallback tính điểm recall cho bước Retrieval.
		"""
		if not chunks:
			return []

		def _neg_confidence(c: Dict[str, Any]) -> float:
			pred = c.get("crag_prediction") or {}
			probs = pred.get("probabilities") or {}
			if "CONTRADICTION/LOSE" in probs:
				return float(probs["CONTRADICTION/LOSE"])
			# Fallback dựa trên crag_score (p_entail càng cao thì phủ định càng yếu)
			return 1.0 - float(c.get("crag_score", 0.0))

		# Sắp xếp tăng dần theo độ tin cậy phủ định (thấp nhất lên trước)
		sorted_chunks = sorted(chunks, key=_neg_confidence)
		fallback_items = []
		for c in sorted_chunks[:top_n]:
			item = dict(c)
			item["is_fallback"] = True
			item["fallback_reason"] = "least_confident_negative"
			fallback_items.append(item)
		return fallback_items

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
				'processed_chunks': List[Dict[str, Any]],   # Chunks cho QA
				'qa_chunks': List[Dict[str, Any]],          # Chunks dành cho phase QA (chứa Tavily)
				'retrieval_chunks': List[Dict[str, Any]],   # Chunks nội bộ dành cho Retrieval metrics
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
		qa_chunks = []

		logger.info(f"[CRAG] Query: '{query}' | Overall Action: {overall_action.upper()}")

		# Bước 2: Phân nhánh hành động (Triggering Actions)
		if overall_action == "correct":
			# Chỉ giữ các chunks đạt "correct" hoặc có điểm cao, sau đó tinh lọc câu
			valid_internal = [c for c in evaluated_chunks if c.get("crag_action") != "incorrect"]
			if not valid_internal:
				valid_internal = self._get_least_confident_negative_chunks(evaluated_chunks, top_n=3)
			internal_refined_chunks = self.refiner.refine_chunks(query, valid_internal, min_score_threshold=l_th)
			qa_chunks = internal_refined_chunks

		elif overall_action == "incorrect":
			# Tài liệu nội bộ sai lệch -> Kích hoạt Web Search (Tavily) cho QA
			logger.info("[CRAG] Kích hoạt Web Search (Tavily) hỗ trợ phase QA do tài liệu nội bộ không liên quan...")
			mcp_res = self.web_search.retrieve(query, top_k=self.mcp_top_k)
			external_chunks = mcp_res.get("context_chunks", [])
			for c in external_chunks:
				c["is_external_search"] = True
				c["source"] = c.get("source") or "tavily_web_search"

			# Với bước Retrieval (tính recall/hit rate trên corpus nội bộ):
			# Fallback về top 3 chunks nội bộ mà NLI ít chắc chắn phủ định nhất
			internal_refined_chunks = self._get_least_confident_negative_chunks(evaluated_chunks, top_n=3)

			# Với phase QA: sử dụng kết quả từ Web Search (kèm fallback nội bộ nếu cần)
			qa_chunks = external_chunks if external_chunks else internal_refined_chunks

		elif overall_action == "ambiguous":
			# Tài liệu nội bộ phân vân -> Kết hợp tài liệu nội bộ đã lọc + Web Search cho QA
			logger.info("[CRAG] Kết hợp Internal Knowledge + Web Search (Tavily)...")
			valid_internal = [c for c in evaluated_chunks if c.get("crag_action") != "incorrect"]
			if not valid_internal:
				valid_internal = self._get_least_confident_negative_chunks(evaluated_chunks, top_n=3)
			internal_refined_chunks = self.refiner.refine_chunks(query, valid_internal, min_score_threshold=l_th)

			mcp_res = self.web_search.retrieve(query, top_k=self.mcp_top_k)
			external_chunks = mcp_res.get("context_chunks", [])
			for c in external_chunks:
				c["is_external_search"] = True
				c["source"] = c.get("source") or "tavily_web_search"

			# Kết hợp cả hai nguồn tri thức cho phase QA
			qa_chunks = internal_refined_chunks + external_chunks

		return {
			"query": query,
			"action": overall_action,
			"processed_chunks": qa_chunks,          # Chunks đầy đủ cho LLM QA (chứa Tavily khi cần)
			"qa_chunks": qa_chunks,                  # Chunks phục vụ phase QA
			"retrieval_chunks": internal_refined_chunks, # CHỈ chunks nội bộ (KHÔNG LƯU TAVILY, dùng tính Recall)
			"internal_chunks": internal_refined_chunks,
			"external_chunks": external_chunks,      # Chunks Tavily bổ trợ ngoài
			"evaluation_details": evaluated_chunks,
		}
