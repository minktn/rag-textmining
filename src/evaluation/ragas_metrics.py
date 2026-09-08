"""
RAGAS Metrics — LLM-as-Judge Evaluation (Multi-Provider Support)
================================================================
Đánh giá chất lượng RAG ở mức Generation & Context thông qua LLM-as-a-judge:
- Faithfulness: Mức độ câu trả lời được hỗ trợ bởi ngữ cảnh (không bịa đặt).
- Answer Relevancy: Mức độ câu trả lời giải quyết trực tiếp câu hỏi.
- Context Precision: Tỷ lệ thông tin hữu ích trong ngữ cảnh được truy xuất.
- Context Recall: Mức độ ngữ cảnh bao quát đầy đủ thông tin của Ground Truth.

Hỗ trợ luân phiên 3 dịch vụ LLM thông qua `RAGAS_SERVICE` trong `settings.py`:
  1. "nvidia": ChatOpenAI qua NVIDIA NIM Endpoint (mặc định: settings.NVIDIA_LLM)
  2. "groq": ChatGroq qua Groq API (mặc định: settings.GROQ_LLM)
  3. "google" / "gemini": ChatGoogleGenerativeAI qua Google GenAI (mặc định: settings.GEMINI_LLM)
"""

import logging
import math
import os
import sys
import time
import types
from typing import Any, Dict, List, Optional

# Compatibility shim for ragas importing deprecated langchain_community.chat_models.vertexai
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        from langchain_google_vertexai import ChatVertexAI
        vertexai_shim.ChatVertexAI = ChatVertexAI
    except Exception:
        vertexai_shim.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_shim

from src.config import settings
from .text_processing import safe_print

print = safe_print

logger = logging.getLogger(__name__)


class RagasJudge:
    """OOP Evaluator sử dụng RAGAS framework hỗ trợ luân phiên 3 dịch vụ LLM Judge (NVIDIA, Groq, Google)."""

    def __init__(
        self,
        service: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        rate_limit_rps: float = 0.5,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        self.max_workers = max_workers or getattr(settings, "RAGAS_MAX_WORKERS", 4)
        self.batch_size = batch_size if batch_size is not None else getattr(settings, "RAGAS_BATCH_SIZE", self.max_workers)

        # Mặc định lấy dịch vụ từ settings.RAGAS_SERVICE hoặc settings.LLM_SERVICE
        self.service = (
            service
            or getattr(settings, "RAGAS_SERVICE", None)
            or getattr(settings, "LLM_SERVICE", "nvidia")
            or "nvidia"
        ).lower().strip()

        if self.service == "gemini":
            self.service = "google"

        # Khởi tạo API key & Model theo service tương tự như LLMManager
        if self.service == "nvidia":
            self.api_key = api_key or getattr(settings, "NVIDIA_KEY", None) or os.getenv("NVIDIA_KEY")
            self.base_url = base_url or getattr(settings, "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
            self.model_name = (
                model_name
                or getattr(settings, "RAGAS_LLM", None)
                or getattr(settings, "NVIDIA_LLM", "nvidia/nemotron-3-ultra-550b-a55b")
            )
        elif self.service == "groq":
            self.api_key = api_key or getattr(settings, "GROQ_KEY", None) or os.getenv("GROQ_API_KEY")
            self.base_url = None
            self.model_name = (
                model_name
                or getattr(settings, "RAGAS_LLM", None)
                or getattr(settings, "GROQ_LLM", "llama-3.3-70b-versatile")
            )
        elif self.service in ("google", "gemini"):
            self.service = "google"
            self.api_key = api_key or getattr(settings, "GEMINI_KEY", None) or os.getenv("GOOGLE_API_KEY")
            self.base_url = None
            self.model_name = (
                model_name
                or getattr(settings, "RAGAS_LLM", None)
                or getattr(settings, "GEMINI_LLM", "gemma-4-31b-it")
            )
        else:
            raise ValueError(
                f"Dịch vụ RAGAS_SERVICE '{self.service}' không hợp lệ. Vui lòng chọn 'nvidia', 'groq', hoặc 'google'."
            )

        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.rate_limit_rps = rate_limit_rps

    @staticmethod
    def get_default_model(service: str) -> str:
        """Lấy model mặc định cho từng dịch vụ theo settings."""
        s = service.lower()
        if s in ("google", "gemini"):
            return getattr(settings, "GEMINI_LLM", "gemma-4-31b-it")
        elif s == "groq":
            return getattr(settings, "GROQ_LLM", "llama-3.3-70b-versatile")
        else:
            return getattr(settings, "NVIDIA_LLM", "nvidia/nemotron-3-ultra-550b-a55b")

    def is_available(self) -> bool:
        """Kiểm tra xem API Key và các thư viện cần thiết đã sẵn sàng chưa."""
        if not self.api_key:
            return False
        try:
            import datasets
            import ragas
            return True
        except ImportError:
            return False

    def _build_langchain_llm(self, rate_limiter: Any) -> Any:
        """Khởi tạo LangChain LLM phù hợp với self.service đã chọn."""
        if self.service == "nvidia":
            from langchain_openai import ChatOpenAI

            os.environ["OPENAI_API_KEY"] = self.api_key
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.4,
                max_tokens=16384,
                rate_limiter=rate_limiter,
                seed=42,
                request_timeout=300,
                max_retries=5,
            )
        elif self.service == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model_name=self.model_name,
                groq_api_key=self.api_key,
                temperature=0.4,
                rate_limiter=rate_limiter,
            )
        elif self.service == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.4,
                rate_limiter=rate_limiter,
            )
        else:
            raise ValueError(f"Dịch vụ LLM '{self.service}' không được hỗ trợ.")

    def _evaluate_single_batch(
        self,
        batch_results: List[Dict[str, Any]],
        batch_idx: int = 1,
        total_batches: int = 1,
        workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Đánh giá 1 batch kết quả qua RAGAS framework không dùng dummy data."""
        if not batch_results:
            return {}

        from datasets import Dataset
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from ragas import evaluate as ragas_evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig

        num_workers = workers or self.max_workers
        is_single_case = (len(batch_results) == 1)
        qid = batch_results[0].get("id", f"case_{batch_idx}") if is_single_case else None

        if is_single_case:
            safe_print(
                f"  [RAGAS Case {batch_idx}/{total_batches}] Đang chấm case '{qid}' "
                f"(workers={num_workers}, service='{self.service}')..."
            )
        else:
            safe_print(
                f"  [RAGAS Batch {batch_idx}/{total_batches}] Đang gửi request LLM Judge "
                f"cho {len(batch_results)} câu hỏi (workers={num_workers}, service='{self.service}')..."
            )

        ragas_data = {
            "question": [r.get("question", "") for r in batch_results],
            "answer": [r.get("generated_answer", "") for r in batch_results],
            "contexts": [r.get("retrieved_contexts", []) for r in batch_results],
            "ground_truth": [r.get("ground_truth", "") for r in batch_results],
        }
        dataset = Dataset.from_dict(ragas_data)

        rate_limiter = InMemoryRateLimiter(requests_per_second=self.rate_limit_rps)
        langchain_llm = self._build_langchain_llm(rate_limiter)
        ragas_llm_wrapper = LangchainLLMWrapper(langchain_llm)

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": settings.DEVICE},
        )

        selected_metrics = [
            faithfulness,
            AnswerRelevancy(strictness=1),
            context_precision,
            context_recall,
        ]

        # Tối ưu timeout và retry để tránh lỗi 503 và TimeoutError khi server NVIDIA bị quá tải
        run_config = RunConfig(
            max_workers=num_workers,
            timeout=300,
            max_retries=5,
            max_wait=120,
        )

        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        try:
            ragas_result = ragas_evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=ragas_llm_wrapper,
                embeddings=embeddings,
                run_config=run_config,
            )

            df = ragas_result.to_pandas()
            for idx, r in enumerate(batch_results):
                missing_metrics = []
                for m in metric_names:
                    if m in df.columns and idx < len(df):
                        val = df.iloc[idx][m]
                        try:
                            f_val = float(val)
                            if not math.isnan(f_val):
                                r[f"ragas_{m}"] = round(f_val, 4)
                            else:
                                r[f"ragas_{m}"] = None
                                missing_metrics.append(m)
                        except (ValueError, TypeError):
                            r[f"ragas_{m}"] = None
                            missing_metrics.append(m)
                    else:
                        r[f"ragas_{m}"] = None
                        missing_metrics.append(m)

                # Không dùng dummy data: nếu lỗi thì để None
                if missing_metrics:
                    for m in missing_metrics:
                        r[f"ragas_{m}"] = None

            if is_single_case:
                r0 = batch_results[0]
                m_parts = [f"{m}={r0.get('ragas_' + m)}" for m in metric_names]
                scores_str = ", ".join(m_parts)
                safe_print(f"  ✓ [RAGAS Case {batch_idx}/{total_batches}] Đã chấm '{qid}': {scores_str}")
            else:
                for r in batch_results:
                    cqid = r.get("id", "case")
                    m_parts = [f"{m}={r.get('ragas_' + m)}" for m in metric_names]
                    scores_str = ", ".join(m_parts)
                    safe_print(f"  ✓ [RAGAS Case] Đã chấm '{cqid}': {scores_str}")
                safe_print(f"  ✓ [RAGAS Block {batch_idx}/{total_batches}] Hoàn tất & lưu đồng thời block {len(batch_results)} câu hỏi.")
            return getattr(ragas_result, "_repr_dict", {})
        except Exception as e:
            logger.error(f"[RAGAS] Lỗi tại Batch {batch_idx}/{total_batches}: {e}")
            safe_print(f"  ✗ [RAGAS Error Batch {batch_idx}/{total_batches}]: {e}")
            for r in batch_results:
                for m in metric_names:
                    if f"ragas_{m}" not in r or r[f"ragas_{m}"] is None:
                        r[f"ragas_{m}"] = None
            return {}

    def evaluate(
        self,
        results: List[Dict[str, Any]],
        metrics_list: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        on_batch_completed: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Tính toán RAGAS metrics cho tập kết quả theo từng batch.
        Cơ chế BẢO LƯU: Giữ nguyên các câu đã có đủ 4 điểm RAGAS hợp lệ,
        CHỈ chạy để FILL bổ sung các câu bị miss/lỗi, và lưu streaming per-batch.
        """
        if not results:
            return {}

        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        def is_valid_ragas(r: Dict[str, Any]) -> bool:
            return all(
                r.get(f"ragas_{m}") is not None
                and not (isinstance(r.get(f"ragas_{m}"), float) and math.isnan(r.get(f"ragas_{m}")))
                for m in metric_names
            )

        def build_ragas_summary(all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            summary: Dict[str, Any] = {}
            for m in metric_names:
                key = f"ragas_{m}"
                vals = [r[key] for r in all_items if r.get(key) is not None]
                summary[m] = round(sum(vals) / len(vals), 4) if vals else None
            return summary

        if not self.api_key:
            logger.warning(f"[RAGAS] Không tìm thấy API Key cho dịch vụ '{self.service}'. Bỏ qua RAGAS.")
            safe_print(f"\n[RAGAS Warning] Chưa cấu hình API Key cho dịch vụ RAGAS '{self.service}' trong .env!")
            return build_ragas_summary(results)

        workers = max_workers or self.max_workers
        bs = batch_size if batch_size is not None else getattr(settings, "RAGAS_BATCH_SIZE", workers)

        # ── 1. Bảo lưu case đã có điểm, chỉ lọc ra các case bị miss ──
        already_valid = [r for r in results if is_valid_ragas(r)]
        pending_items = [r for r in results if not is_valid_ragas(r)]

        if already_valid:
            safe_print(f"\n[RAGAS Bảo Lưu] Phát hiện {len(already_valid)}/{len(results)} câu hỏi ĐÃ CÓ kết quả RAGAS hợp lệ trước đó (giữ nguyên).")

        if not pending_items:
            safe_print(f"  ✓ Toàn bộ {len(results)} câu hỏi đã có kết quả RAGAS đầy đủ! Không cần gọi LLM Judge nữa.")
            return build_ragas_summary(results)

        total_batches = (len(pending_items) + bs - 1) // bs
        safe_print(
            f"\n[RAGAS Concurrency] Đang đánh giá cho {len(pending_items)} câu hỏi chia làm {total_batches} block "
            f"(mỗi block {bs} requests đồng thời, workers={workers}, LLM Judge='{self.service}', model='{self.model_name}')..."
        )

        batches = [pending_items[i : i + bs] for i in range(0, len(pending_items), bs)]

        # ── 2. Đánh giá từng batch và lưu bảo lưu ngay lập tức ─────────
        for idx, batch in enumerate(batches, 1):
            self._evaluate_single_batch(batch, idx, total_batches, workers)

            # Cập nhật và lưu bảo lưu ngay sau mỗi batch
            current_summary = build_ragas_summary(results)
            if on_batch_completed:
                on_batch_completed(batch, current_summary)

            if idx < total_batches:
                time.sleep(2.0)  # Cooldown xả nghẽn API quota

        final_summary = build_ragas_summary(results)
        return final_summary


def compute_ragas_metrics(
    results: List[Dict[str, Any]],
    service: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Hàm helper tương thích ngược."""
    judge = RagasJudge(
        service=service,
        model_name=model_name,
        batch_size=batch_size,
        max_workers=max_workers,
    )
    return judge.evaluate(results, batch_size=batch_size, max_workers=max_workers)

