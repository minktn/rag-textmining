import os
import re
import time
from typing import List, Optional
from openai import OpenAI

from src.configs import settings


class RAGFusionQueryGenerator:
	"""Generator for creating multi-query expansions using NVIDIA endpoint LLM (SUPPORT_LLM)."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		model: Optional[str] = None,
		temperature: float = 0.7,
	):
		self.api_key = api_key or settings.NVIDIA_KEY or os.getenv("NVIDIA_KEY")
		self.base_url = base_url or getattr(settings, "NVIDIA_BASE_URL", None) or "https://integrate.api.nvidia.com/v1"
		self.model = model or settings.SUPPORT_LLM or "openai/gpt-oss-120b"
		self.temperature = temperature
		self._client: Optional[OpenAI] = None

	@property
	def client(self) -> OpenAI:
		if self._client is None:
			if not self.api_key:
				raise ValueError("Không tìm thấy NVIDIA_KEY. Vui lòng thiết lập biến môi trường NVIDIA_KEY.")
			self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
		return self._client

	def generate_queries(
		self,
		original_query: str,
		num_queries: int = 4,
		diverse: bool = False,
		language: str = "vi",
	) -> List[str]:
		"""Sinh ra danh sách các câu truy vấn mở rộng từ câu truy vấn ban đầu.

		Parameters
		----------
		original_query : str
			Câu hỏi truy vấn ban đầu của người dùng.
		num_queries : int, default=4
			Số lượng câu hỏi mở rộng cần sinh ra.
		diverse : bool, default=False
			Nếu True, yêu cầu LLM mở rộng câu hỏi theo các góc độ đa dạng khác nhau.
		language : str, default="vi"
			Ngôn ngữ cho prompt ("vi" hoặc "en").

		Returns
		-------
		List[str]
			Danh sách các truy vấn đã được sinh ra (không bao gồm câu truy vấn gốc).
		"""
		if not original_query or not original_query.strip():
			return []

		if language == "vi":
			if diverse:
				system_prompt = (
					"Bạn là một chuyên gia truy xuất thông tin pháp lý và tra cứu dữ liệu. "
					"Nhiệm vụ của bạn là sinh ra các câu truy vấn tìm kiếm đa dạng nhằm khai thác "
					"các khía cạnh khác nhau của câu hỏi từ người dùng. "
					"Mỗi câu truy vấn nên khai thác một góc độ: dùng từ đồng nghĩa, mở rộng/thu hẹp phạm vi, "
					"và đề cập tới các khái niệm pháp lý liên quan. Tránh sinh các câu lặp lại."
				)
				user_prompt = f"Hãy sinh đúng {num_queries} câu truy vấn tìm kiếm đa dạng cho câu hỏi: '{original_query}'\nChỉ trả về danh sách các câu truy vấn, mỗi câu trên 1 dòng."
			else:
				system_prompt = (
					"Bạn là trợ lý AI thông minh chuyên sinh các câu truy vấn tìm kiếm tìm kiếm thông tin liên quan."
				)
				user_prompt = f"Hãy sinh {num_queries} câu truy vấn tìm kiếm liên quan đến câu hỏi: '{original_query}'\nChỉ trả về danh sách các câu truy vấn, mỗi câu trên 1 dòng."
		else:
			if diverse:
				system_prompt = (
					"You are a search expert. Generate diverse search queries that explore different aspects "
					"of the user's question. Each query should target a different angle: use synonyms, vary specificity, "
					"and consider related sub-topics."
				)
				user_prompt = f"Generate {num_queries} diverse search queries for: '{original_query}'\nReturn only the list of queries, one per line."
			else:
				system_prompt = "You are a helpful assistant that generates multiple search queries based on a single input query."
				user_prompt = f"Generate {num_queries} search queries related to: '{original_query}'\nReturn only the list of queries, one per line."

		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]

		try:
			response = self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				temperature=self.temperature,
			)
			content = response.choices[0].message.content or ""
			return self._parse_queries(content, num_queries)
		except Exception as e:
			print(f"[RAG-Fusion Error] Lỗi khi gọi NVIDIA LLM: {e}")
			return []

	def _parse_queries(self, content: str, max_queries: int) -> List[str]:
		"""Tách các dòng và làm sạch danh sách câu hỏi sinh ra."""
		lines = content.strip().split("\n")
		queries = []
		for line in lines:
			cleaned = line.strip()
			if not cleaned:
				continue
			# Loại bỏ số thứ tự ở đầu dòng (ví dụ: "1. ", "1)", "- ", "* ")
			cleaned = re.sub(r"^[\d\.\-\*\)\s]+", "", cleaned).strip()
			# Loại bỏ dấu ngoặc kép bọc ngoài nếu có
			cleaned = cleaned.strip('"\'')
			if cleaned and cleaned not in queries:
				queries.append(cleaned)
			if len(queries) >= max_queries:
				break
		return queries
