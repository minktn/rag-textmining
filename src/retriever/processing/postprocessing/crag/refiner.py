"""
CRAG Knowledge Refiner (Document Decomposition & Knowledge Striping)
=====================================================================
Thực hiện tinh lọc tri thức (Knowledge Refinement) theo cơ chế CRAG:
1. Phân rã văn bản thành các dải tri thức nhỏ (Knowledge Strips / Sentences).
2. Lọc bỏ các câu gây nhiễu hoặc không liên quan dựa trên điểm Evaluator.
3. Tái cấu trúc thành ngữ cảnh súc tích, cô đọng cho LLM.
"""

import re
from typing import Any, Dict, List, Optional

from .evaluator import BamiBERTRetrievalEvaluator


class KnowledgeRefiner:
	"""Module tinh lọc và tái cấu trúc tri thức cho CRAG."""

	def __init__(
		self,
		evaluator: Optional[BamiBERTRetrievalEvaluator] = None,
		min_strip_chars: int = 20,
		keep_top_strips: int = 5,
	):
		self.evaluator = evaluator or BamiBERTRetrievalEvaluator()
		self.min_strip_chars = min_strip_chars
		self.keep_top_strips = keep_top_strips

	def split_into_strips(self, text: str) -> List[str]:
		"""Tách một đoạn văn bản thành các dải tri thức (câu hoặc khoản nhỏ)."""
		if not text:
			return []

		# Tách theo dấu câu tiếng Việt hoặc xuống dòng
		raw_sentences = re.split(r"(?<=[.!?;\n])\s+", text.strip())
		strips = [s.strip() for s in raw_sentences if len(s.strip()) >= self.min_strip_chars]
		return strips if strips else [text.strip()]

	def refine_chunk(
		self,
		query: str,
		chunk: Dict[str, Any],
		min_score_threshold: float = 0.35,
	) -> Dict[str, Any]:
		"""Tinh lọc một chunk: tách câu, chấm điểm từng câu và giữ lại các câu phù hợp."""
		content = chunk.get("content", "")
		strips = self.split_into_strips(content)

		if len(strips) <= 1:
			return chunk

		scores = self.evaluator.evaluate_batch(query, strips)
		scored_strips = list(zip(strips, scores))

		# Lọc các strip có điểm >= min_score_threshold
		valid_strips = [s for s, sc in scored_strips if sc >= min_score_threshold]

		if not valid_strips:
			# Nếu không câu nào vượt ngưỡng, giữ lại câu có điểm cao nhất
			best_strip = max(scored_strips, key=lambda x: x[1])[0]
			valid_strips = [best_strip]

		refined_chunk = dict(chunk)
		refined_chunk["content"] = " ".join(valid_strips)
		refined_chunk["is_refined"] = True
		refined_chunk["num_original_strips"] = len(strips)
		refined_chunk["num_refined_strips"] = len(valid_strips)
		return refined_chunk

	def refine_chunks(
		self,
		query: str,
		chunks: List[Dict[str, Any]],
		min_score_threshold: float = 0.35,
	) -> List[Dict[str, Any]]:
		"""Tinh lọc toàn bộ danh sách chunks."""
		refined = []
		for chunk in chunks:
			refined.append(self.refine_chunk(query, chunk, min_score_threshold=min_score_threshold))
		return refined
