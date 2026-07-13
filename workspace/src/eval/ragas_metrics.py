"""
RAGAS Metrics — LLM-as-Judge Evaluation
=========================================
Tính RAGAS metrics: faithfulness, answer_relevancy,
context_precision, context_recall.

Sử dụng ChatGroq làm LLM judge và HuggingFaceEmbeddings.
"""

from src.configs import settings


def compute_ragas_metrics(results: list[dict]) -> dict:
    """Tính RAGAS metrics cho list kết quả RAG.

    Args:
        results: List kết quả từ RAGEvaluator.evaluate_all(),
                 mỗi item chứa question, generated_answer, retrieved_contexts, ground_truth.

    Returns:
        dict chứa aggregate RAGAS scores.
        Trả về {} nếu dependencies chưa cài hoặc gặp lỗi.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"\n⚠️  RAGAS dependencies chưa được cài đặt: {e}")
        print("   Chạy: pip install ragas langchain-groq langchain-huggingface datasets")
        return {}

    print("\n🔬 Đang tính RAGAS metrics (sử dụng LLM-as-judge)...")

    # ── Chuẩn bị RAGAS dataset ──────────────────────────────────
    ragas_data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': [],
    }

    for r in results:
        ragas_data['question'].append(r['question'])
        ragas_data['answer'].append(r['generated_answer'])
        ragas_data['contexts'].append(r['retrieved_contexts'])
        ragas_data['ground_truth'].append(r['ground_truth'])

    dataset = Dataset.from_dict(ragas_data)

    # ── Setup LLM + Embeddings cho RAGAS ────────────────────────
    llm = ChatGroq(
        model=settings.TEST_LLM,
        api_key=settings.GROQ_API_KEY,
        temperature=0.0,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={'device': settings.DEVICE},
    )

    # ── Chạy RAGAS evaluation ───────────────────────────────────
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    try:
        ragas_result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as e:
        print(f"\n⚠️  RAGAS evaluation gặp lỗi: {e}")
        print("   Có thể do rate limit Groq API. Thử giảm --limit hoặc đợi.")
        return {}

    # ── Trích xuất kết quả ──────────────────────────────────────
    ragas_scores = {}

    # Aggregate scores (Ragas lưu điểm trung bình trong thuộc tính nội bộ _repr_dict)
    scores_dict = getattr(ragas_result, '_repr_dict', {})
    for metric_name in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        val = scores_dict.get(metric_name, None)
        if val is not None:
            ragas_scores[metric_name] = round(float(val), 4)

    # Per-question scores (từ DataFrame nếu có)
    try:
        df = ragas_result.to_pandas()
        for idx, r in enumerate(results):
            for metric_name in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                col_name = metric_name
                if col_name in df.columns and idx < len(df):
                    val = df.iloc[idx][col_name]
                    r[f'ragas_{metric_name}'] = round(float(val), 4) if val is not None else None
    except Exception:
        pass  # Per-question extraction failed, aggregate still available

    return ragas_scores
