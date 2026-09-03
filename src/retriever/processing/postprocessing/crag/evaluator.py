"""
CRAG Retrieval Evaluator (BamiBERT-ViLegalNLI)
==============================================
Chịu trách nhiệm đánh giá độ tin cậy P(Entailment) của các tài liệu truy xuất nội bộ
đối với câu hỏi pháp luật của người dùng để phân loại hành động (Correct / Ambiguous / Incorrect).

YÊU CẦU BẮT BUỘC:
-----------------
Pipeline CRAG BẮT BUỘC phải có FastAPI NLI Server (port 8001) đang chạy trước khi thực thi.
Khởi động NLI server qua lệnh:
    python -m src.models.nli_model.api
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config import settings

logger = logging.getLogger(__name__)


def verify_nli_service(api_base_url: Optional[str] = None, timeout: float = 5.0) -> bool:
	"""Kiểm tra xem NLI API Server (port 8001) có đang hoạt động hay không."""
	base_url = (api_base_url or getattr(settings, "NLI_API_URL", "http://localhost:8001")).rstrip("/")
	health_url = f"{base_url}/health"
	try:
		resp = requests.get(health_url, timeout=timeout)
		return resp.status_code == 200
	except requests.exceptions.RequestException:
		return False


class BamiBERTRetrievalEvaluator:
	"""Evaluator đánh giá mức độ liên quan của tài liệu dựa trên BamiBERT ViLegalNLI qua FastAPI Server."""

	def __init__(
		self,
		api_url: Optional[str] = None,
		upper_threshold: float = 0.75,
		lower_threshold: float = 0.35,
		timeout: float = 30.0,
		auto_verify: bool = True,
	):
		base_url = (api_url or getattr(settings, "NLI_API_URL", "http://localhost:8001")).rstrip("/")
		self.api_url = f"{base_url}/predict_batch" if not base_url.endswith("/predict_batch") else base_url
		self.upper_threshold = upper_threshold
		self.lower_threshold = lower_threshold
		self.timeout = timeout

		if auto_verify and not verify_nli_service(base_url, timeout=3.0):
			logger.warning(
				f"[CRAG Warning] NLI API Server tại '{base_url}' chưa phản hồi. "
				f"Vui lòng đảm bảo đã chạy: 'python -m src.models.nli_model.api'"
			)

	def evaluate_chunks(
		self,
		query: str,
		chunks: List[Dict[str, Any]],
		upper_th: Optional[float] = None,
		lower_th: Optional[float] = None,
	) -> Tuple[str, List[Dict[str, Any]]]:
		"""
		Đánh giá danh sách chunks qua NLI API và xác định overall_action ('correct' | 'ambiguous' | 'incorrect').
		
		Raises:
			ConnectionError: Khi NLI API server không phản hồi.
		"""
		if not chunks:
			return "incorrect", []

		u_th = upper_th if upper_th is not None else self.upper_threshold
		l_th = lower_th if lower_th is not None else self.lower_threshold

		items = [
			{
				"question": query,
				"context": chunk.get("content", ""),
				"id": str(chunk.get("id", idx)),
			}
			for idx, chunk in enumerate(chunks)
		]
		payload = {
			"items": items,
			"metadata": {"pipeline": "CRAG", "source": "crag_retrieval_evaluator"},
		}

		predictions = []
		try:
			resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
			if resp.status_code == 200:
				data = resp.json()
				predictions = data.get("predictions") or []
			else:
				raise RuntimeError(f"HTTP {resp.status_code}")
		except Exception as e:
			logger.warning(
				f"[CRAG Warning] NLI Server (port 8001) không phản hồi ({e}). "
				f"Tự động chuyển sang nạp NLIEngine trực tiếp nội bộ..."
			)
			try:
				from src.models.nli_model.api import NLIEngine
				if not hasattr(self, "_local_nli_engine") or self._local_nli_engine is None:
					self._local_nli_engine = NLIEngine()
				pairs = [(item["question"], item["context"]) for item in items]
				predictions = self._local_nli_engine.predict_batch(pairs)
			except Exception as local_err:
				raise ConnectionError(
					f"[CRAG Error] Không thể kết nối tới NLI API Server tại '{self.api_url}' "
					f"và cũng không thể nạp mô hình cục bộ ({local_err})."
				) from local_err

		evaluated_chunks: List[Dict[str, Any]] = []
		for chunk, pred in zip(chunks, predictions):
			p_entail = pred["probabilities"]["ENTAILMENT/WIN"]
			if p_entail >= u_th:
				chunk_action = "correct"
			elif p_entail <= l_th:
				chunk_action = "incorrect"
			else:
				chunk_action = "ambiguous"

			c = dict(chunk)
			c["crag_score"] = p_entail
			c["crag_action"] = chunk_action
			c["crag_prediction"] = pred
			evaluated_chunks.append(c)

		actions = [c["crag_action"] for c in evaluated_chunks]
		if "correct" in actions:
			overall_action = "correct"
		elif "ambiguous" in actions:
			overall_action = "ambiguous"
		else:
			overall_action = "incorrect"

		return overall_action, evaluated_chunks

	def evaluate_single(self, query: str, document: str) -> float:
		"""Tính điểm xác suất P(Entailment) cho 1 cặp (query, document)."""
		_, eval_chunks = self.evaluate_chunks(query, [{"content": document}])
		return eval_chunks[0].get("crag_score", 0.0) if eval_chunks else 0.0

	def evaluate_batch(self, query: str, documents: List[str]) -> List[float]:
		"""Tính điểm xác suất cho danh sách documents dạng text."""
		chunks = [{"content": doc} for doc in documents]
		_, eval_chunks = self.evaluate_chunks(query, chunks)
		return [c.get("crag_score", 0.0) for c in eval_chunks]
