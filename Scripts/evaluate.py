"""
RAG Evaluation Script — Luật Đất đai 2024
==========================================
CLI entrypoint cho RAG evaluation pipeline.

Cách chạy:
    python -m scripts.evaluate                        # Standard RAG
    python -m scripts.evaluate --graph                # Chạy GraphRAG
    python -m scripts.evaluate --rag-fusion           # Chạy RAG-Fusion
    python -m scripts.evaluate --graph --rag-fusion   # Kết hợp GraphRAG & RAG-Fusion
    python -m scripts.evaluate --limit 10 --skip-ragas  # Test nhanh 10 câu

Module structure (src/eval/):
    - text_processing: Chuẩn hóa & tokenize tiếng Việt
    - metrics: Basic metrics (F1, BLEU-1, ROUGE-L, Recall@K, Precision@K, nDCG)
    - ragas_metrics: RAGAS framework (LLM-as-judge)
    - evaluator: RAGEvaluator pipeline
    - reporting: Console output & JSON persistence
    - data_loader: Load eval dataset
"""

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
	sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
	sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.database import DBManager
from src.database.embedder import DenseEmbedder
from src.evaluation import (
	MetricsCalculator,
	RAGEvaluator,
	compute_ragas_metrics,
	load_eval_dataset,
	print_summary_table,
	save_results,
)
from src.evaluation.metrics import HAS_ROUGE
from src.evaluation.text_processing import HAS_UNDERTHESEA, safe_print
from src.generation import LLMManager


class TeeLogger:
	def __init__(self, filepath, original_print):
		self.terminal = original_print
		self.file = open(filepath, "w", encoding="utf-8")

	def print(self, *args, **kwargs):
		self.terminal(*args, **kwargs)
		sep = kwargs.get("sep", " ")
		end = kwargs.get("end", "\n")
		text = sep.join(map(str, args)) + end
		self.file.write(text)
		self.file.flush()

	def close(self):
		self.file.close()


log_file_path = Path(__file__).resolve().parent / "evaluate.log"
tee_logger = TeeLogger(log_file_path, safe_print)
print = tee_logger.print


def parse_args():
	parser = argparse.ArgumentParser(
		description="Đánh giá RAG pipeline Luật Đất đai 2024"
	)
	parser.add_argument(
		"--eval-file", type=str, default=None,
		help="Đường dẫn đến file eval JSON (mặc định: eval_landlaw_2024.json)"
	)
	parser.add_argument(
		"--limit", type=int, default=None,
		help="Giới hạn số câu hỏi đánh giá (mặc định: tất cả)"
	)
	parser.add_argument(
		"--skip-ragas", action="store_true",
		help="Bỏ qua RAGAS metrics (tiết kiệm API calls)"
	)
	parser.add_argument(
		"--model", type=str, default=None,
		help="Model LLM sử dụng (mặc định: theo cấu hình LLM_SERVICE trong settings.py)"
	)
	parser.add_argument(
		"--top-k", type=int, default=5,
		help="Số lượng contexts retrieve (mặc định: 5)"
	)
	parser.add_argument(
		"--collection", type=str, default="landlaw",
		help="Tên collection Qdrant (mặc định: landlaw)"
	)
	parser.add_argument(
		"--seed", type=int, default=42,
		help="Random seed cho sampling câu hỏi (mặc định: 42)"
	)
	parser.add_argument(
		"--random", action="store_true",
		help="Bật random sampling câu hỏi thay vì lấy từ đầu"
	)
	parser.add_argument(
		"--retriever-mode", type=str, default=None, choices=["base", "contriever", "graph"],
		help="Chế độ retriever: base | contriever | graph"
	)
	parser.add_argument(
		"--llm-service", type=str, default=None, choices=["nvidia", "groq", "google", "local"],
		help="Dịch vụ LLM chính: nvidia | groq | google | local"
	)
	parser.add_argument(
		"--sub-llm-service", type=str, default=None, choices=["local", "nvidia", "groq", "google"],
		help="Dịch vụ Sub-LLM: local | nvidia | groq | google"
	)
	parser.add_argument(
		"--advanced", type=str, default=None,
		help="Phương thức truy xuất nâng cao (ví dụ: rag_fusion)"
	)
	parser.add_argument(
		"--preprocessing", nargs="*", default=None,
		help="Danh sách phương thức tiền xử lý (ví dụ: hyde)"
	)
	parser.add_argument(
		"--postprocessing", nargs="*", default=None,
		help="Danh sách phương thức hậu xử lý (ví dụ: prompt_compression)"
	)
	parser.add_argument(
		"--ragas-service", type=str, default=None, choices=["nvidia", "groq", "google"],
		help="Dịch vụ LLM Judge cho RAGAS: nvidia | groq | google (mặc định theo settings.RAGAS_SERVICE)"
	)
	parser.add_argument(
		"--graph", action="store_true",
		help="Bật chế độ đánh giá Microsoft GraphRAG pipeline (tương đương --retriever-mode graph)"
	)
	parser.add_argument(
		"--graph-method", type=str, default="local", choices=["local", "global", "drift", "basic"],
		help="Phương thức truy vấn GraphRAG (mặc định: local)"
	)
	return parser.parse_args()


def main():
	args = parse_args()

	# ── Resolve paths & params ────────────────────────────────
	if args.eval_file:
		eval_file = Path(args.eval_file)
	else:
		eval_file = settings.EVAL_DATA_DIR / "eval_landlaw_2024.json"

	retriever_mode = args.retriever_mode or ("graph" if args.graph else "base")

	evaluator = RAGEvaluator(
		retriever_mode=retriever_mode,
		llm_service=args.llm_service,
		sub_llm_service=args.sub_llm_service,
		advanced=args.advanced,
		preprocessing=args.preprocessing,
		postprocessing=args.postprocessing,
		ragas_service=args.ragas_service,
		model_name=args.model,
		collection_name=args.collection,
		top_k=args.top_k,
		graph_method=args.graph_method,
	)

	meta = evaluator.get_pipeline_metadata()
	config_info = meta["configuration"]

	print("=" * 60)
	print("  [RAG EVALUATION] - Land Law 2024")
	print("=" * 60)
	print(f"  Retriever:   {config_info['retriever_mode']}")
	print(f"  Advanced:    {config_info['advanced_method'] or 'None'}")
	print(f"  Preprocess:  {config_info['preprocessing'] or 'None'}")
	print(f"  Postprocess: {config_info['postprocessing'] or 'None'}")
	print(f"  LLM Service: {config_info['llm_service']} ({config_info['llm_model']})")
	print(f"  Sub-LLM:     {config_info['sub_llm_service']} ({config_info['sub_llm_model']})")
	print(f"  RAGAS Judge: {config_info['ragas_service']} ({config_info['ragas_model']})")
	print(f"  Top-K:       {config_info['top_k']}")
	print(f"  Collection:  {config_info['collection_name']}")
	print(f"  Embedding:   {config_info['embedding_model']}")
	print(f"  RAGAS:       {'Skipped' if args.skip_ragas else 'Enabled'}")
	print(f"  Limit:       {args.limit or 'All'}")
	print(f"  Random:      {'Enabled' if args.random else 'Disabled'} (Seed: {args.seed})")
	if meta.get("notice"):
		print(f"  ALERT:       {meta['notice']}")
	print("=" * 60)

	evaluator.run(
		eval_file=eval_file,
		limit=args.limit,
		random_sample=args.random,
		seed=args.seed,
		skip_ragas=args.skip_ragas,
		save=True,
	)

	print("\nEvaluation completed!")
	tee_logger.close()


if __name__ == '__main__':
	main()
