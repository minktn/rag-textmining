"""
LLM Reranker — Large Language Model Hard-Sample Reranker
=========================================================
Bước 2 trong mô hình Filter-then-Rerank (EMNLP 2023):
Đóng vai trò Reranker chuyên biệt cho các mẫu khó (Hard Samples) được bàn giao từ SLMFilter.

Ràng buộc cấu hình:
- Sử dụng SubLLMManager cấu hình cứng với dịch vụ Cloud "nvidia" (không dùng local SLM)
- Model mặc định: settings.NVIDIA_LLM (nvidia/nemotron-3-ultra-550b-a55b)
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.config import settings
from src.generation.sub_llm_manager import SubLLMManager

logger = logging.getLogger(__name__)


class LLMReranker:
    """Bộ xếp hạng lại các mẫu khó (Hard Samples) bằng Large Language Model (NVIDIA)."""

    def __init__(
        self,
        sub_llm_manager: Optional[SubLLMManager] = None,
        model_name: Optional[str] = None,
    ):
        # Cấu hình cứng dịch vụ NVIDIA cho Large LLM Reranker theo yêu cầu
        if sub_llm_manager is not None and sub_llm_manager.service != "local":
            self.sub_llm = sub_llm_manager
        else:
            self.sub_llm = SubLLMManager(service="nvidia")

        self.model_name = model_name or self.sub_llm.get_default_model(self.sub_llm.service)
        logger.info(
            f"[LLMReranker] Khởi tạo Reranker với service='{self.sub_llm.service}', "
            f"model='{self.model_name}'"
        )

    def score_hard_chunk(self, query: str, chunk_text: str) -> float:
        """
        Gửi đoạn văn bản khó tới Large LLM để chấm điểm mức độ liên quan (0.0 - 1.0).
        """
        prompt = (
            f"Bạn là chuyên gia thẩm định và xếp hạng văn bản pháp luật Việt Nam.\n"
            f"Hãy đánh giá mức độ liên quan và giá trị pháp lý của đoạn văn bản sau "
            f"đối với câu hỏi được cung cấp trên thang điểm từ 0 đến 10 "
            f"(trong đó: 0 = hoàn toàn không liên quan, 10 = trả lời trực tiếp và chính xác nhất).\n\n"
            f"CÂU HỎI:\n{query}\n\n"
            f"ĐOẠN VĂN BẢN:\n{chunk_text[:1500]}\n\n"
            f"Hãy trả về điểm số duy nhất dưới dạng số thực từ 0 đến 10 (Ví dụ: 8.5):"
        )

        try:
            raw_response = self.sub_llm.generate_response(
                prompt=prompt,
                model_name=self.model_name,
                service=self.sub_llm.service,
            )
            if not raw_response:
                return 0.5

            # Trích xuất số thực đầu tiên trong câu trả lời
            match = re.search(r"(\d+(\.\d+)?)", raw_response.strip())
            if match:
                score_10 = float(match.group(1))
                # Giới hạn trong khoảng [0, 10]
                score_10 = max(0.0, min(10.0, score_10))
                return round(score_10 / 10.0, 4)
            return 0.5
        except Exception as e:
            logger.error(f"[LLMReranker] Lỗi khi chấm điểm mẫu khó qua LLM: {e}")
            return 0.5

    def rerank_hard_samples(
        self,
        query: str,
        hard_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Chấm điểm và tái xếp hạng danh sách các mẫu khó.
        """
        if not hard_chunks:
            return []

        reranked_chunks: List[Dict[str, Any]] = []
        for chunk in hard_chunks:
            content = chunk.get("content") or chunk.get("text") or ""
            score = self.score_hard_chunk(query, content)

            chunk_copy = dict(chunk)
            chunk_copy["llm_rerank_score"] = score
            chunk_copy["final_score"] = score
            chunk_copy["source"] = "slm_filter_llm_reranked"
            reranked_chunks.append(chunk_copy)

        # Sắp xếp giảm dần theo điểm số rerank
        reranked_chunks.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
        return reranked_chunks
