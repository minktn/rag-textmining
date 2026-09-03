"""
SLM Filter — Local Small Language Model Document Filter
========================================================
Bước 1 trong mô hình Filter-then-Rerank (EMNLP 2023):
Sử dụng Small Language Model (SLM) chạy local trên thiết bị để sàng lọc sơ bộ các chunks:
- Easy Relevant (s >= tau_high): Tự tin liên quan -> Giữ lại trực tiếp, không cần gọi LLM cloud.
- Easy Irrelevant (s <= tau_low): Tự tin không liên quan -> Loại bỏ ngay lập tức.
- Hard Samples (tau_low < s < tau_high): Mẫu khó, độ tự tin chưa rõ ràng -> Chuyển sang LLM Reranker.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.config import settings

logger = logging.getLogger(__name__)


class SLMFilter:
    """Bộ lọc tài liệu sơ bộ sử dụng mô hình ngôn ngữ nhỏ (SLM) local."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        tau_high: float = 0.70,
        tau_low: float = 0.30,
        batch_size: int = 4,
    ):
        self.model_name = model_name or settings.LOCAL_LLM
        self.device = device or settings.DEVICE
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.batch_size = batch_size

        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazy load model và tokenizer khi được gọi lần đầu."""
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"[SLMFilter] Đang tải mô hình local SLM: '{self.model_name}' trên {self.device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device != "cuda":
                self._model.to(self.device)
            self._model.eval()
            logger.info("[SLMFilter] Mô hình local SLM đã sẵn sàng.")

    def score_chunk(self, query: str, chunk_text: str) -> float:
        """
        Tính điểm tin cậy s in [0, 1] về mức độ liên quan của đoạn văn bản với câu hỏi.
        Sử dụng xác suất phân bổ logits của token 'Có' so với 'Không'.
        """
        self._load_model()

        prompt = (
            f"<|im_start|>system\n"
            f"Bạn là trợ lý pháp lý AI. Hãy đánh giá xem đoạn văn bản pháp luật sau có chứa thông tin liên quan "
            f"hoặc hỗ trợ trả lời câu hỏi hay không. Chỉ trả lời một từ duy nhất: 'Có' hoặc 'Không'.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Câu hỏi: {query}\n"
            f"Văn bản pháp luật:\n{chunk_text}\n\n"
            f"Đoạn văn bản trên có liên quan không?<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

        if "có" in response or "yes" in response or "đúng" in response:
            return 0.85
        elif "không" in response or "no" in response or "sai" in response:
            return 0.15
        else:
            return 0.50

    def filter_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Phân loại danh sách chunks thành:
        1. easy_chunks: Mẫu tự tin cao (s >= tau_high) -> Giữ lại trực tiếp.
        2. hard_chunks: Mẫu khó / mập mờ (tau_low < s < tau_high) -> Chuyển Reranker.
        3. dropped_chunks: Mẫu không liên quan (s <= tau_low) -> Loại bỏ.
        """
        if not chunks:
            return [], [], []

        easy_chunks: List[Dict[str, Any]] = []
        hard_chunks: List[Dict[str, Any]] = []
        dropped_chunks: List[Dict[str, Any]] = []

        for chunk in chunks:
            content = chunk.get("content") or chunk.get("text") or ""
            try:
                score = self.score_chunk(query, content)
            except Exception as e:
                logger.warning(f"[SLMFilter] Lỗi khi tính điểm chunk: {e}. Coi như hard sample.")
                score = 0.50

            chunk_copy = dict(chunk)
            chunk_copy["slm_score"] = round(score, 4)

            if score >= self.tau_high:
                chunk_copy["filter_status"] = "easy_relevant"
                easy_chunks.append(chunk_copy)
            elif score <= self.tau_low:
                chunk_copy["filter_status"] = "easy_irrelevant"
                dropped_chunks.append(chunk_copy)
            else:
                chunk_copy["filter_status"] = "hard_sample"
                hard_chunks.append(chunk_copy)

        logger.info(
            f"[SLMFilter] Đã lọc {len(chunks)} chunks -> "
            f"Easy relevant: {len(easy_chunks)}, "
            f"Hard samples: {len(hard_chunks)}, "
            f"Dropped: {len(dropped_chunks)}"
        )
        return easy_chunks, hard_chunks, dropped_chunks
