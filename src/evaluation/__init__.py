"""
src.evaluation — RAG Evaluation Module
=======================================
Module đánh giá toàn diện cho hệ thống RAG Hỏi đáp Luật Đất đai 2024.

Kiến trúc OOP:
    - RAGEvaluator: Orchestrator điều phối toàn bộ quy trình đánh giá.
    - MetricsCalculator: Tính các chỉ số Retrieval (Recall@K, Precision@K, MRR, nDCG)
      và Generation (F1, Exact Match, BLEU-1, ROUGE-L).
    - RagasJudge: Đánh giá LLM-as-a-judge (Faithfulness, Answer Relevancy, Context Precision/Recall).
    - EvaluationReporter: In bảng tổng hợp console và lưu kết quả JSON.
    - EvalDataLoader: Load và sample bộ dữ liệu câu hỏi đánh giá.
"""

from .text_processing import normalize_text, tokenize_vi, safe_print
from .metrics import MetricsCalculator
from .ragas_metrics import RagasJudge, compute_ragas_metrics
from .evaluator import RAGEvaluator
from .reporting import EvaluationReporter, print_summary_table, save_results
from .data_loader import EvalDataLoader, load_eval_dataset

__all__ = [
    # OOP Classes
    "RAGEvaluator",
    "MetricsCalculator",
    "RagasJudge",
    "EvaluationReporter",
    "EvalDataLoader",
    # Functions / Helpers
    "normalize_text",
    "tokenize_vi",
    "safe_print",
    "compute_ragas_metrics",
    "print_summary_table",
    "save_results",
    "load_eval_dataset",
]
