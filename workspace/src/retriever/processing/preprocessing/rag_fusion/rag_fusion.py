from typing import Any, Callable, Dict, List, Optional, Tuple

from .fusion import reciprocal_rank_fusion
from .query_generator import RAGFusionQueryGenerator


class RAGFusionProcessor:
	"""Pipeline xử lý RAG-Fusion kết hợp Query Expansion và Reciprocal Rank Fusion."""

	def __init__(
		self,
		query_generator: Optional[RAGFusionQueryGenerator] = None,
		k: int = 60,
	):
		self.query_generator = query_generator or RAGFusionQueryGenerator()
		self.k = k

	def expand_queries(
		self,
		query: str,
		num_queries: int = 4,
		diverse: bool = False,
		language: str = "vi",
	) -> List[str]:
		"""Tạo tập hợp các câu hỏi truy vấn gồm câu truy vấn gốc + các câu truy vấn được mở rộng.

		Returns
		-------
		List[str]
			Danh sách chứa [original_query, query_1, query_2, ...]
		"""
		generated_queries = self.query_generator.generate_queries(
			original_query=query,
			num_queries=num_queries,
			diverse=diverse,
			language=language,
		)
		return [query] + generated_queries

	def process(
		self,
		query: str,
		retrieve_fn: Callable[[str], List[Any]],
		num_queries: int = 4,
		diverse: bool = False,
		original_query_weight: float = 3.0,
		top_k: Optional[int] = None,
		language: str = "vi",
	) -> Tuple[List[str], List[Dict[str, Any]]]:
		"""Thực thi quy trình RAG-Fusion đầy đủ:
		1. Sinh các câu hỏi mở rộng từ câu hỏi gốc.
		2. Gọi hàm retrieve_fn cho từng câu hỏi để lấy danh sách tài liệu.
		3. Áp dụng thuật toán Reciprocal Rank Fusion (RRF) để tổng hợp điểm số.

		Parameters
		----------
		query : str
			Câu hỏi ban đầu.
		retrieve_fn : Callable[[str], List[Any]]
			Hàm tìm kiếm/truy xuất nhận vào query string và trả về danh sách tài liệu.
		num_queries : int, default=4
			Số câu hỏi mở rộng cần sinh thêm.
		diverse : bool, default=False
			Mở rộng câu hỏi theo các góc độ đa dạng.
		original_query_weight : float, default=3.0
			Trọng số ưu tiên cho câu hỏi gốc khi tính RRF.
		top_k : Optional[int], default=None
			Số lượng tài liệu tối đa cần trả về sau khi xếp hạng lại.
		language : str, default="vi"
			Ngôn ngữ gợi ý cho LLM ("vi" hoặc "en").

		Returns
		-------
		Tuple[List[str], List[Dict[str, Any]]]
			(danh_sách_các_câu_hỏi, danh_sách_tài_liệu_đã_được_rff_rerank)
		"""
		all_queries = self.expand_queries(
			query=query,
			num_queries=num_queries,
			diverse=diverse,
			language=language,
		)

		all_results: Dict[str, List[Any]] = {}
		for q in all_queries:
			try:
				all_results[q] = retrieve_fn(q)
			except Exception as e:
				print(f"[RAG-Fusion Warning] Lỗi khi truy xuất cho query '{q}': {e}")
				all_results[q] = []

		query_weights = {query: original_query_weight} if original_query_weight != 1.0 else None

		fused_results = reciprocal_rank_fusion(
			search_results_dict=all_results,
			k=self.k,
			query_weights=query_weights,
		)

		if top_k is not None:
			fused_results = fused_results[:top_k]

		return all_queries, fused_results
