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

from src.configs import settings
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
		help=f"Model LLM sử dụng (mặc định: {settings.TEST_LLM})"
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
		"--graph", action="store_true",
		help="Bật chế độ đánh giá Microsoft GraphRAG pipeline"
	)
	parser.add_argument(
		"--graph-method", type=str, default="local", choices=["local", "global", "drift", "basic"],
		help="Phương thức truy vấn GraphRAG (mặc định: local)"
	)
	parser.add_argument(
		"--rag-fusion", action="store_true",
		help="Bật/tắt RAG-Fusion (Multi-Query Expansion + RRF)"
	)
	return parser.parse_args()


def main():
	args = parse_args()

	# ── Resolve paths & params ────────────────────────────────
	if args.eval_file:
		eval_file = Path(args.eval_file)
	else:
		eval_file = settings.EVAL_DATA_DIR / "eval_landlaw_2024.json"

	model_name = args.model or settings.TEST_LLM
	top_k = args.top_k
	collection = args.collection
	seed = args.seed

	mode_str = []
	if args.graph:
		mode_str.append(f"GraphRAG ({args.graph_method})")
	if getattr(args, "rag_fusion", False):
		mode_str.append("RAG-Fusion")
	if not mode_str:
		mode_str.append("Standard RAG")

	print("=" * 60)
	print("  [RAG EVALUATION] - Land Law 2024")
	print("=" * 60)
	print(f"  Mode:        {' + '.join(mode_str)}")
	print(f"  Model:       {model_name}")
	print(f"  Top-K:       {top_k}")
	print(f"  Collection:  {collection}")
	print(f"  Embedding:   {settings.EMBEDDING_MODEL}")
	print(f"  Tokenizer:   {'Underthesea' if HAS_UNDERTHESEA else 'Simple split'}")
	print(f"  ROUGE-L:     {'rouge-score' if HAS_ROUGE else 'LCS fallback'}")
	print(f"  RAGAS:       {'Skipped' if args.skip_ragas else 'Enabled'}")
	print(f"  Limit:       {args.limit or 'All'}")
	print(f"  Random Sampling: {'Enabled' if args.random else 'Disabled'} (Seed: {seed})")
	print("=" * 60)

	# ── Load data ─────────────────────────────────────────────
	eval_metadata, questions = load_eval_dataset(eval_file, limit=None)

	if not questions:
		print("No questions to evaluate!")
		sys.exit(1)

	if args.random:
		import random
		random.seed(seed)
		if args.limit and args.limit < len(questions):
			questions = random.sample(questions, args.limit)
			print(f"  [Random Sampling] Sampled {len(questions)} questions with seed={seed}")
	elif args.limit and args.limit < len(questions):
		questions = questions[:args.limit]

	# ── Init components ───────────────────────────────────────
	print("\nInitializing components...")
	embedder = DenseEmbedder(settings.EMBEDDING_MODEL)

	db_manager = DBManager(
		url=settings.QDRANT_URL,
		api_key=settings.QDRANT_API_KEY
	)

	llm_manager = LLMManager(
		api_key=settings.GROQ_API_KEY,
		temperature=0.1
	)

	# ── Phase 1: Run RAG pipeline ─────────────────────────────
	print(f"\nPhase 1: Running RAG pipeline on {len(questions)} questions...\n")

	evaluator = RAGEvaluator(
		embedder=embedder,
		db_manager=db_manager,
		llm_manager=llm_manager,
		model_name=model_name,
		collection_name=collection,
		top_k=top_k,
		use_graph=args.graph,
		use_rag_fusion=args.rag_fusion,
		graph_method=args.graph_method,
	)
	results = evaluator.evaluate_all(questions)

	# ── Phase 2: Compute basic metrics ────────────────────────
	print("\nPhase 2: Computing basic metrics...")
	basic_metrics = MetricsCalculator.aggregate(results, top_k=top_k)

	# ── Phase 3: RAGAS metrics (optional) ─────────────────────
	ragas_scores = {}
	if not args.skip_ragas:
		print("\nPhase 3: Computing RAGAS metrics...")
		ragas_scores = compute_ragas_metrics(results)
	else:
		print("\nPhase 3: RAGAS metrics — SKIPPED")

	# ── Phase 4: Report & Save ────────────────────────────────
	print_summary_table(basic_metrics, ragas_scores, model_name)

	save_results(
		results=results,
		basic_metrics=basic_metrics,
		ragas_scores=ragas_scores,
		eval_metadata=eval_metadata,
		model_name=model_name,
		top_k=top_k,
		output_dir=settings.EVAL_RESULTS_DIR,
	)

	print("\nEvaluation completed!")
	tee_logger.close()


if __name__ == '__main__':
	main()
