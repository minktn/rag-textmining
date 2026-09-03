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
import os
import sys
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
    ):
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
                temperature=0.0,
                max_tokens=16384,
                rate_limiter=rate_limiter,
                seed=42,
            )
        elif self.service == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model_name=self.model_name,
                groq_api_key=self.api_key,
                temperature=0.0,
                rate_limiter=rate_limiter,
            )
        elif self.service == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.0,
                rate_limiter=rate_limiter,
            )
        else:
            raise ValueError(f"Dịch vụ LLM '{self.service}' không được hỗ trợ.")

    def evaluate(
        self,
        results: List[Dict[str, Any]],
        metrics_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Tính toán RAGAS metrics cho tập kết quả sử dụng dịch vụ LLM đã cấu hình.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Danh sách kết quả evaluation, mỗi item chứa:
            question, generated_answer, retrieved_contexts, ground_truth.
        metrics_list : Optional[List[str]]
            Danh sách metric muốn chạy (mặc định: cả 4 metrics).
        """
        if not results:
            return {}

        if not self.api_key:
            logger.warning(f"[RAGAS] Không tìm thấy API Key cho dịch vụ '{self.service}'. Bỏ qua RAGAS.")
            print(f"\n[RAGAS Warning] Chưa cấu hình API Key cho dịch vụ RAGAS '{self.service}' trong .env!")
            return {}

        try:
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
        except ImportError as e:
            logger.warning(f"[RAGAS] Thiếu thư viện phụ thuộc: {e}")
            print(f"\n[RAGAS Warning] Thư viện RAGAS chưa sẵn sàng: {e}")
            return {}

        print(
            f"\n[RAGAS] Đang đánh giá {len(results)} câu hỏi với LLM Judge: "
            f"service='{self.service}', model='{self.model_name}'..."
        )

        # 1. Chuẩn bị dataset cho RAGAS
        ragas_data = {
            "question": [r.get("question", "") for r in results],
            "answer": [r.get("generated_answer", "") for r in results],
            "contexts": [r.get("retrieved_contexts", []) for r in results],
            "ground_truth": [r.get("ground_truth", "") for r in results],
        }
        dataset = Dataset.from_dict(ragas_data)

        # 2. Khởi tạo LLM Judge & Embeddings
        rate_limiter = InMemoryRateLimiter(requests_per_second=self.rate_limit_rps)
        langchain_llm = self._build_langchain_llm(rate_limiter)
        ragas_llm_wrapper = LangchainLLMWrapper(langchain_llm)

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": settings.DEVICE},
        )

        # 3. Chọn metrics
        selected_metrics = [
            faithfulness,
            AnswerRelevancy(strictness=1),
            context_precision,
            context_recall,
        ]

        run_config = RunConfig(
            max_workers=2,
            timeout=180,
            max_retries=10,
            max_wait=60,
        )

        # 4. Thực thi RAGAS
        try:
            ragas_result = ragas_evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=ragas_llm_wrapper,
                embeddings=embeddings,
                run_config=run_config,
            )
        except Exception as e:
            logger.error(f"[RAGAS] Lỗi khi chạy ragas_evaluate: {e}")
            print(f"\n[RAGAS Error] Đánh giá thất bại: {e}")
            return {}

        # 5. Tổng hợp điểm số
        ragas_scores: Dict[str, float] = {}
        scores_dict = getattr(ragas_result, "_repr_dict", {})
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        for m in metric_names:
            val = scores_dict.get(m)
            if val is not None:
                ragas_scores[m] = round(float(val), 4)

        # Gán kết quả chi tiết từng câu hỏi vào results
        try:
            df = ragas_result.to_pandas()
            for idx, r in enumerate(results):
                for m in metric_names:
                    if m in df.columns and idx < len(df):
                        val = df.iloc[idx][m]
                        r[f"ragas_{m}"] = round(float(val), 4) if val is not None else None
        except Exception as ex:
            logger.warning(f"[RAGAS] Không thể trích xuất per-question scores: {ex}")

        return ragas_scores


def compute_ragas_metrics(
    results: List[Dict[str, Any]],
    service: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, float]:
    """Hàm helper tương thích ngược."""
    judge = RagasJudge(service=service, model_name=model_name)
    return judge.evaluate(results)
