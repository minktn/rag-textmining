"""
RAG Evaluation Script — Luật Đất đai 2024
==========================================
CLI entrypoint cho RAG evaluation pipeline.

Cách chạy:
    python -m scripts.evaluate                        # Full 150 câu
    python -m scripts.evaluate --limit 10             # 10 câu đầu
    python -m scripts.evaluate --limit 10 --skip-ragas  # Chỉ basic metrics
    python -m scripts.evaluate --model llama-3.3-70b-versatile --top-k 3

Module structure (src/eval/):
    - text_processing: Chuẩn hóa & tokenize tiếng Việt
    - metrics: Basic metrics (F1, BLEU-1, ROUGE-L, Recall@K, Precision@K, nDCG)
    - ragas_metrics: RAGAS framework (LLM-as-judge)
    - evaluator: RAGEvaluator pipeline
    - reporting: Console output & JSON persistence
    - data_loader: Load eval dataset
"""

import os
import argparse
import sys
from pathlib import Path

# Setup Hugging Face mirror and custom cache paths to avoid Windows TLS socket errors (WinError 10054/10038)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["FASTEMBED_CACHE_PATH"] = str(Path.home() / ".cache" / "fastembed")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import settings
from src.data_pipeline import DenseEmbedder
from src.database import DBManager
from src.llm import LLMManager

from src.eval import (
    RAGEvaluator,
    MetricsCalculator,
    compute_ragas_metrics,
    print_summary_table,
    save_results,
    load_eval_dataset,
)
from src.eval.text_processing import HAS_UNDERTHESEA, safe_print
from src.eval.metrics import HAS_ROUGE

print = safe_print


def parse_args():
    parser = argparse.ArgumentParser(
        description='Đánh giá RAG pipeline Luật Đất đai 2024'
    )
    parser.add_argument(
        '--eval-file', type=str, default=None,
        help='Đường dẫn đến file eval JSON (mặc định: eval_landlaw_2024.json)'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Giới hạn số câu hỏi đánh giá (mặc định: tất cả)'
    )
    parser.add_argument(
        '--skip-ragas', action='store_true',
        help='Bỏ qua RAGAS metrics (tiết kiệm API calls)'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help=f'Model LLM sử dụng (mặc định: {settings.TEST_LLM})'
    )
    parser.add_argument(
        '--top-k', type=int, default=5,
        help='Số lượng contexts retrieve (mặc định: 5)'
    )
    parser.add_argument(
        '--collection', type=str, default=settings.COLLECTION_NAME,
        help=f'Tên collection Qdrant (mặc định: {settings.COLLECTION_NAME})'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths & params ────────────────────────────────
    if args.eval_file:
        eval_file = Path(args.eval_file)
    else:
        eval_file = settings.EVAL_DATA_DIR / 'eval_landlaw_2024.json'

    model_name = args.model or settings.TEST_LLM
    top_k = args.top_k
    collection = args.collection

    print("=" * 60)
    print("  [RAG EVALUATION] - Land Law 2024")
    print("=" * 60)
    print(f"  Model:       {model_name}")
    print(f"  Top-K:       {top_k}")
    print(f"  Collection:  {collection}")
    print(f"  Embedding:   {settings.EMBEDDING_MODEL}")
    print(f"  Tokenizer:   {'Underthesea' if HAS_UNDERTHESEA else 'Simple split (Warning: pip install underthesea)'}")
    print(f"  ROUGE-L:     {'rouge-score' if HAS_ROUGE else 'LCS fallback'}")
    print(f"  RAGAS:       {'Skipped' if args.skip_ragas else 'Enabled'}")
    print(f"  Limit:       {args.limit or 'All'}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    eval_metadata, questions = load_eval_dataset(eval_file, limit=args.limit)

    if not questions:
        print("No questions to evaluate!")
        sys.exit(1)

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

    from src.retriever import Retriever
    retriever = Retriever(
        db_manager=db_manager,
        embedder=embedder,
        collection_name=collection,
        dense_candidate_limit=settings.RETRIEVAL_CANDIDATE_LIMIT,
        rerank_limit=top_k,
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
        retriever=retriever,
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


if __name__ == '__main__':
    main()
