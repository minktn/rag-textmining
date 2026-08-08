import os
import time
from groq import Groq
from openai import OpenAI

from src.configs import settings


class LLMManager:
	# Rate limiting: 30 RPM -> 1 request every 2.0 seconds minimum delay
	_last_request_time = 0.0
	_min_delay = 2.0

	def __init__(self, api_key=None, temperature=0.1):
		self.groq_api_key = api_key or settings.GROQ_API_KEY
		self.nvidia_api_key = settings.NVIDIA_KEY or os.getenv("NVIDIA_KEY")
		self.nvidia_base_url = getattr(settings, "NVIDIA_BASE_URL", None) or "https://integrate.api.nvidia.com/v1"
		self.temperature = temperature

		self._groq_client = None
		self._nvidia_client = None

		self.system_prompt = (
			"Bạn là trợ lý AI chuyên nghiệp. "
			"Hãy trả lời bằng tiếng Việt "
			"dựa trên ngữ cảnh được cung cấp."
		)

	@property
	def groq_client(self):
		if self._groq_client is None:
			self._groq_client = Groq(api_key=self.groq_api_key)
		return self._groq_client

	@property
	def nvidia_client(self):
		if self._nvidia_client is None:
			if not self.nvidia_api_key:
				raise ValueError("NVIDIA_KEY chưa được cấu hình trong .env")
			self._nvidia_client = OpenAI(api_key=self.nvidia_api_key, base_url=self.nvidia_base_url)
		return self._nvidia_client

	def construct_prompt(self, query, docs=None):
		if docs:
			context = "\n\n".join(
				self._format_doc(doc, index)
				for index, doc in enumerate(docs, start=1)
			)
			return f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{query}"
		return f"CÂU HỎI:\n{query}"

	def _format_doc(self, doc, index):
		metadata = doc.get("metadata") or doc.get("payload") or {}
		lines = [f"Tài liệu {index}:"]

		for key, label in (
			("source", "Nguồn"),
			("chapter", "Chương"),
			("section", "Mục"),
			("article", "Điều"),
			("article_no", "Số điều"),
			("chapter_no", "Số chương"),
			("section_no", "Số mục"),
			("clause_nos", "Các khoản"),
			("ref_article_nos", "Điều được viện dẫn"),
		):
			value = metadata.get(key)
			if value not in (None, "", []):
				lines.append(f"{label}: {value}")

		if doc.get("dense_score") is not None:
			lines.append(f"Điểm truy xuất dense: {doc['dense_score']}")
		if doc.get("rerank_score") is not None:
			lines.append(f"Điểm xếp hạng lại: {doc['rerank_score']}")
		if doc.get("rrf_score") is not None:
			lines.append(f"Điểm RRF: {doc['rrf_score']}")
		if doc.get("source"):
			lines.append(f"Nguồn truy xuất: {doc['source']}")

		lines.append("Nội dung:")
		lines.append(str(doc.get("content", "")))
		return "\n".join(lines)

	def generate_response(self, prompt, model_name=None):
		target_model = model_name or settings.TEST_LLM

		# Enforce rate limit delay
		now = time.time()
		elapsed = now - LLMManager._last_request_time
		if elapsed < LLMManager._min_delay:
			time.sleep(LLMManager._min_delay - elapsed)
		LLMManager._last_request_time = time.time()

		try:
			messages = [
				{"role": "system", "content": self.system_prompt},
				{"role": "user", "content": prompt},
			]

			if target_model.startswith("openai/") or "nvidia" in target_model.lower():
				response = self.nvidia_client.chat.completions.create(
					model=target_model,
					messages=messages,
					temperature=self.temperature,
				)
			else:
				response = self.groq_client.chat.completions.create(
					model=target_model,
					messages=messages,
					temperature=self.temperature,
				)
			return response.choices[0].message.content

		except Exception as e:
			print(f"Lỗi khi tạo câu trả lời ({target_model}): {e}")
			return None
