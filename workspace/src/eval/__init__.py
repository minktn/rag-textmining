"""
src.eval — RAG Evaluation Module
=================================
Module đánh giá pipeline RAG cho Luật Đất đai 2024.

Cấu trúc:
    - text_processing: Chuẩn hóa & tokenize tiếng Việt (Underthesea)
    - metrics: Basic metrics (F1, BLEU-1, ROUGE-L, Recall@K, Precision@K, nDCG, ...)
    - ragas_metrics: RAGAS framework (LLM-as-judge)
    - evaluator: RAGEvaluator — orchestrate pipeline
    - reporting: In bảng tổng hợp & lưu kết quả JSON
    - data_loader: Load eval dataset
"""

from src.eval.text_processing import normalize_text, tokenize_vi
from src.eval.metrics import MetricsCalculator
from src.eval.ragas_metrics import compute_ragas_metrics
from src.eval.evaluator import RAGEvaluator
from src.eval.reporting import print_summary_table, save_results
from src.eval.data_loader import load_eval_dataset

__all__ = [
    'normalize_text',
    'tokenize_vi',
    'MetricsCalculator',
    'compute_ragas_metrics',
    'RAGEvaluator',
    'print_summary_table',
    'save_results',
    'load_eval_dataset',
]
