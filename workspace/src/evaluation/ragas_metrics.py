"""
RAGAS Metrics — LLM-as-Judge Evaluation (NVIDIA Endpoint + GLM-5.2)
===================================================================
Tính RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall.

Sử dụng ChatOpenAI kết nối tới NVIDIA API Endpoint với RAGAS_LLM ('z-ai/glm-5.2')
và HuggingFaceEmbeddings cho embedding evaluation.
"""

import os
import sys
from typing import Any, Dict, List

from src.configs import settings


def compute_ragas_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
	"""Tính RAGAS metrics cho list kết quả RAG bằng LLM-as-judge qua NVIDIA API.

	Parameters
	----------
	results : List[Dict[str, Any]]
		List kết quả từ RAGEvaluator.evaluate_all(), mỗi item chứa:
		question, generated_answer, retrieved_contexts, ground_truth.

	Returns
	-------
	Dict[str, float]
		Dictionary chứa các chỉ số RAGAS trung bình (faithfulness, answer_relevancy, ...).
		Trả về {} nếu dependencies chưa cài hoặc gặp lỗi.
	"""
	try:
		from datasets import Dataset
		from langchain_community.embeddings import HuggingFaceEmbeddings
		from langchain_core.rate_limiters import InMemoryRateLimiter
		from langchain_openai import ChatOpenAI
		from ragas import evaluate as ragas_evaluate
		from ragas.llms import LangchainLLMWrapper
		from ragas.metrics import AnswerRelevancy, context_precision, context_recall, faithfulness
		from ragas.run_config import RunConfig

		answer_relevancy = AnswerRelevancy(strictness=1)
	except ImportError as e:
		print(f"\n[RAGAS Warning] RAGAS dependencies missing: {e}")
		print("   Vui lòng cài đặt: pip install ragas langchain-openai langchain-community datasets")
		return {}

	nvidia_key = settings.NVIDIA_KEY or os.getenv("NVIDIA_KEY") or os.getenv("NVIDIA_API_KEY")
	if not nvidia_key:
		print("\n[RAGAS Error] Không tìm thấy NVIDIA_KEY trong .env để chạy RAGAS LLM judge!")
		return {}

	nvidia_base_url = getattr(settings, "NVIDIA_BASE_URL", None) or "https://integrate.api.nvidia.com/v1"
	ragas_model_name = getattr(settings, "RAGAS_LLM", None) or "z-ai/glm-5.2"

	print(f"\nComputing RAGAS metrics using NVIDIA LLM-as-Judge ({ragas_model_name})...")

	# ── 1. Chuẩn bị dataset ──────────────────────────────────────────
	ragas_data = {
		"question": [],
		"answer": [],
		"contexts": [],
		"ground_truth": [],
	}

	for r in results:
		ragas_data["question"].append(r.get("question", ""))
		ragas_data["answer"].append(r.get("generated_answer", ""))
		ragas_data["contexts"].append(r.get("retrieved_contexts", []))
		ragas_data["ground_truth"].append(r.get("ground_truth", ""))

	dataset = Dataset.from_dict(ragas_data)

	# ── 2. Khởi tạo LLM & Embeddings cho RAGAS ──────────────────────
	# InMemoryRateLimiter tránh gửi quá dồn dập
	rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)

	# Đặt biến môi trường tạm thời cho OpenAI client nếu cần
	os.environ["OPENAI_API_KEY"] = nvidia_key

	langchain_llm = ChatOpenAI(
		model=ragas_model_name,
		api_key=nvidia_key,
		base_url=nvidia_base_url,
		temperature=0.0,
		max_tokens=16384,
		rate_limiter=rate_limiter,
		model_kwargs={"seed": 42},
	)
	ragas_llm_wrapper = LangchainLLMWrapper(langchain_llm)

	embeddings = HuggingFaceEmbeddings(
		model_name=settings.EMBEDDING_MODEL,
		model_kwargs={"device": settings.DEVICE},
	)

	# ── 3. Cấu hình Metrics & RunConfig ─────────────────────────────
	metrics = [
		faithfulness,
		answer_relevancy,
		context_precision,
		context_recall,
	]

	run_config = RunConfig(
		max_workers=2,
		timeout=180,
		max_retries=10,
		max_wait=60,
	)

	# ── 4. Chạy RAGAS Evaluation ────────────────────────────────────
	try:
		ragas_result = ragas_evaluate(
			dataset=dataset,
			metrics=metrics,
			llm=ragas_llm_wrapper,
			embeddings=embeddings,
			run_config=run_config,
		)
	except Exception as e:
		print(f"\n[RAGAS Error] RAGAS evaluation failed: {e}")
		return {}

	# ── 5. Trích xuất & Tổng hợp chỉ số ─────────────────────────────
	ragas_scores: Dict[str, float] = {}

	scores_dict = getattr(ragas_result, "_repr_dict", {})
	for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
		val = scores_dict.get(metric_name, None)
		if val is not None:
			ragas_scores[metric_name] = round(float(val), 4)

	# Gán kết quả chi tiết từng câu hỏi vào results
	try:
		df = ragas_result.to_pandas()
		for idx, r in enumerate(results):
			for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
				if metric_name in df.columns and idx < len(df):
					val = df.iloc[idx][metric_name]
					r[f"ragas_{metric_name}"] = round(float(val), 4) if val is not None else None
	except Exception as ex:
		print(f"[RAGAS Warning] Không thể trích xuất per-question scores: {ex}")

	return ragas_scores
