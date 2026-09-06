import os
import time
from typing import Optional
from groq import Groq
from openai import OpenAI

from src.config import settings


class LLMManager:
	# Rate limiting: 30 RPM -> 1 request every 2.0 seconds minimum delay
	_last_request_time = 0.0
	_min_delay = 2.0

	def __init__(
		self,
		api_key: Optional[str] = None,
		service: Optional[str] = None,
		mode: Optional[str] = None,
	):
		# Mặc định lấy dịch vụ từ settings.LLM_SERVICE
		self.service = (service or getattr(settings, "LLM_SERVICE", "nvidia") or "nvidia").lower()

		# Khởi tạo API key & URL từ settings
		self.groq_api_key = api_key if self.service == "groq" else (settings.GROQ_KEY or api_key)
		self.nvidia_api_key = (api_key if self.service == "nvidia" else None) or settings.NVIDIA_KEY or api_key
		self.nvidia_base_url = settings.NVIDIA_BASE_URL
		self.gemini_key = (api_key if self.service in ("google", "gemini") else None) or settings.GEMINI_KEY or api_key

		# Chế độ chạy: "base" hoặc "reason" (mặc định lấy từ settings.LLM_MODE)
		self.mode = (mode or getattr(settings, "LLM_MODE", "base") or "base").lower()

		# Cấu hình temperature và max_tokens từ settings theo mode
		self.temperature: float = (
			settings.REASONING_TEMP if self.mode in ("reason", "reasoning") else settings.BASE_TEMP
		)
		self.max_tokens: int = (
			settings.REASONING_MAX_TOKENS if self.mode in ("reason", "reasoning") else settings.BASE_MAX_TOKENS
		)

		self._groq_client = None
		self._nvidia_client = None
		self._google_client = None
		self._local_pipeline = None

		self.system_prompt = (
			"Bạn là trợ lý luật sư AI. "
			"Hãy trả lời bằng tiếng Việt "
			"dựa trên các thông tin được cung cấp."
		)

	@property
	def groq_client(self):
		if self._groq_client is None:
			if not self.groq_api_key:
				raise ValueError("GROQ_KEY chưa được cấu hình trong .env hoặc settings")
			self._groq_client = Groq(api_key=self.groq_api_key, timeout=getattr(settings, "LLM_TIMEOUT", 60.0))
		return self._groq_client

	@property
	def nvidia_client(self):
		if self._nvidia_client is None:
			if not self.nvidia_api_key:
				raise ValueError("NVIDIA_KEY chưa được cấu hình trong .env hoặc settings")
			self._nvidia_client = OpenAI(
				api_key=self.nvidia_api_key,
				base_url=self.nvidia_base_url,
				timeout=getattr(settings, "LLM_TIMEOUT", 60.0),
			)
		return self._nvidia_client

	@property
	def google_client(self):
		if self._google_client is None:
			if not self.gemini_key:
				raise ValueError("GEMINI_KEY chưa được cấu hình trong .env hoặc settings")
			from langchain_google_genai import ChatGoogleGenerativeAI
			self._google_client = ChatGoogleGenerativeAI(
				model=settings.GEMINI_LLM,
				google_api_key=self.gemini_key,
				temperature=self.temperature,
				max_output_tokens=self.max_tokens,
				timeout=getattr(settings, "LLM_TIMEOUT", 60),
				max_retries=3,
			)
		return self._google_client

	@property
	def local_pipeline(self):
		if self._local_pipeline is None:
			import torch
			from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
			model_id = settings.LOCAL_LLM
			device = settings.DEVICE
			tokenizer = AutoTokenizer.from_pretrained(model_id)
			model = AutoModelForCausalLM.from_pretrained(
				model_id,
				torch_dtype=torch.float16 if device == "cuda" else torch.float32,
				device_map="auto" if device == "cuda" else None,
			)
			self._local_pipeline = pipeline(
				"text-generation",
				model=model,
				tokenizer=tokenizer,
			)
		return self._local_pipeline

	def get_default_model(self, service: str) -> str:
		service = service.lower()
		if service in ("google", "gemini"):
			return settings.GEMINI_LLM
		elif service == "nvidia":
			return settings.NVIDIA_LLM
		elif service == "local":
			return settings.LOCAL_LLM
		else:  # groq
			return settings.GROQ_LLM

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

	def generate_response(
		self,
		prompt: str,
		model_name: Optional[str] = None,
		service: Optional[str] = None,
		mode: Optional[str] = None,
	):
		active_service = (service or self.service).lower()
		active_mode = (mode or self.mode).lower()
		target_model = model_name or self.get_default_model(active_service)

		# Cấu hình temp và max_tokens lấy hoàn toàn theo mode từ settings
		curr_temp = settings.REASONING_TEMP if active_mode in ("reason", "reasoning") else settings.BASE_TEMP
		curr_max_tokens = settings.REASONING_MAX_TOKENS if active_mode in ("reason", "reasoning") else settings.BASE_MAX_TOKENS

		# Enforce rate limit delay
		now = time.time()
		elapsed = now - LLMManager._last_request_time
		if elapsed < LLMManager._min_delay:
			time.sleep(LLMManager._min_delay - elapsed)
		LLMManager._last_request_time = time.time()

		messages = [
			{"role": "system", "content": self.system_prompt},
			{"role": "user", "content": prompt},
		]

		if active_service in ("google", "gemini"):
			from langchain_core.messages import HumanMessage, SystemMessage
			from langchain_google_genai import ChatGoogleGenerativeAI
			llm = ChatGoogleGenerativeAI(
				model=target_model,
				google_api_key=self.gemini_key or settings.GEMINI_KEY,
				temperature=curr_temp,
				max_output_tokens=curr_max_tokens,
				timeout=getattr(settings, "LLM_TIMEOUT", 60),
				max_retries=3,
			)
			try:
				response = llm.invoke([
					SystemMessage(content=self.system_prompt),
					HumanMessage(content=prompt),
				])
			except Exception as e:
				import logging
				logging.getLogger(__name__).warning(f"[LLMManager] Lỗi gọi Google API sau timeout/retry ({e}). Trả về rỗng.")
				return ""

			content = response.content
			if isinstance(content, list):
				parts = []
				for item in content:
					if isinstance(item, str):
						parts.append(item)
					elif isinstance(item, dict):
						if item.get("type") == "thinking" or "thinking" in item:
							continue
						if "text" in item:
							parts.append(str(item["text"]))
					elif hasattr(item, "text"):
						parts.append(str(item.text))
				return " ".join(parts).strip()
			return str(content or "").strip()

		elif active_service == "local":
			pipe = self.local_pipeline
			prompt_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
			res = pipe(
				prompt_text,
				max_new_tokens=curr_max_tokens,
				do_sample=curr_temp > 0,
				temperature=max(curr_temp, 0.01) if curr_temp > 0 else None,
			)
			generated = res[0]["generated_text"]
			if "<|im_start|>assistant\n" in generated:
				return generated.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
			return generated.strip()

		elif active_service == "nvidia":
			response = self.nvidia_client.chat.completions.create(
				model=target_model,
				messages=messages,
				temperature=curr_temp,
				max_tokens=curr_max_tokens,
			)
			return response.choices[0].message.content

		else:  # groq
			response = self.groq_client.chat.completions.create(
				model=target_model,
				messages=messages,
				temperature=curr_temp,
				max_tokens=curr_max_tokens,
			)
			return response.choices[0].message.content




