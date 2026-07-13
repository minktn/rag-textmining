"""
Reporting — Console Output & JSON Persistence
================================================
In bảng tổng hợp kết quả ra console và lưu results/summary ra JSON files.
"""

import json
from datetime import datetime
from pathlib import Path

from src.eval.text_processing import HAS_UNDERTHESEA


def print_summary_table(basic_metrics: dict, ragas_scores: dict, model_name: str):
    """In bảng tổng hợp kết quả ra console."""
    sep = "═" * 60
    thin_sep = "─" * 60

    print(f"\n{sep}")
    print(f"  📊  KẾT QUẢ ĐÁNH GIÁ RAG — Luật Đất đai 2024")
    print(f"  Model: {model_name}")
    top_k = basic_metrics.get('top_k', '?')
    print(f"  Top-K: {top_k}")
    print(f"{sep}\n")

    # ── Retrieval Metrics ─────────────────────────────────
    print(f"  {'RETRIEVAL METRICS':^56}")
    print(f"  {thin_sep}")
    print(f"  {'Metric':<35} {'Value':>18}")
    print(f"  {thin_sep}")
    print(f"  {'Hit Rate':<35} {basic_metrics.get('retrieval_hit_rate', 0):.2%}")
    print(f"  {'MRR (Mean Reciprocal Rank)':<35} {basic_metrics.get('mrr', 0):.4f}")
    print(f"  {f'Recall@{top_k}':<35} {basic_metrics.get('avg_recall_at_k', 0):.4f}")
    print(f"  {f'Precision@{top_k}':<35} {basic_metrics.get('avg_precision_at_k', 0):.4f}")
    print(f"  {'nDCG':<35} {basic_metrics.get('avg_ndcg', 0):.4f}")
    print(f"  {'Avg Retrieval Latency':<35} {basic_metrics.get('avg_retrieval_latency_ms', 0):.1f} ms")
    print()

    # ── Generation Metrics ────────────────────────────────
    print(f"  {'GENERATION METRICS':^56}")
    print(f"  {thin_sep}")
    print(f"  {'Metric':<35} {'Value':>18}")
    print(f"  {thin_sep}")
    print(f"  {'Exact Match Rate':<35} {basic_metrics.get('exact_match_rate', 0):.2%}")
    print(f"  {'Avg F1 Score':<35} {basic_metrics.get('avg_f1', 0):.4f}")
    print(f"  {'Avg BLEU-1':<35} {basic_metrics.get('avg_bleu_1', 0):.4f}")
    print(f"  {'Avg ROUGE-L':<35} {basic_metrics.get('avg_rouge_l', 0):.4f}")
    print(f"  {'Avg Generation Latency':<35} {basic_metrics.get('avg_generation_latency_ms', 0):.1f} ms")
    print(f"  {'Avg E2E Latency':<35} {basic_metrics.get('avg_e2e_latency_ms', 0):.1f} ms")
    print()

    # ── RAGAS Metrics ─────────────────────────────────────
    if ragas_scores:
        print(f"  {'RAGAS METRICS (LLM-as-Judge)':^56}")
        print(f"  {thin_sep}")
        print(f"  {'Metric':<35} {'Value':>18}")
        print(f"  {thin_sep}")
        for metric, value in ragas_scores.items():
            label = metric.replace('_', ' ').title()
            print(f"  {label:<35} {value:.4f}")
        print()

    # ── Per Question Type ─────────────────────────────────
    per_type = basic_metrics.get('per_question_type', {})
    if per_type:
        print(f"  {'BREAKDOWN BY QUESTION TYPE':^56}")
        print(f"  {thin_sep}")
        print(f"  {'Type':<14} {'N':>4} {'Hit%':>7} {'Rcl@K':>7} {'F1':>7} {'BLEU':>7} {'RG-L':>7}")
        print(f"  {thin_sep}")
        for qt, stats in per_type.items():
            print(
                f"  {qt:<14} {stats['count']:>4}"
                f" {stats['hit_rate']:>6.1%}"
                f" {stats['avg_recall_at_k']:>7.4f}"
                f" {stats['avg_f1']:>7.4f}"
                f" {stats['avg_bleu_1']:>7.4f}"
                f" {stats['avg_rouge_l']:>7.4f}"
            )
        print()

    print(f"{sep}")


def save_results(
    results: list[dict],
    basic_metrics: dict,
    ragas_scores: dict,
    eval_metadata: dict,
    model_name: str,
    top_k: int,
    output_dir: Path,
):
    """Lưu kết quả chi tiết + summary ra JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Detailed results ──────────────────────────────────
    # Clean up payloads (not JSON serializable if they contain special types)
    clean_results = []
    for r in results:
        cr = {k: v for k, v in r.items() if k != 'retrieved_payloads'}
        clean_results.append(cr)

    detail_path = output_dir / f'eval_results_{timestamp}.json'
    detail_data = {
        'metadata': {
            'timestamp': timestamp,
            'model': model_name,
            'top_k': top_k,
            'total_questions': len(results),
            'eval_source': eval_metadata,
            'tokenizer': 'underthesea' if HAS_UNDERTHESEA else 'simple_split',
        },
        'results': clean_results,
    }
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(detail_data, f, ensure_ascii=False, indent=4)

    # ── Summary ───────────────────────────────────────────
    summary_path = output_dir / f'eval_summary_{timestamp}.json'
    summary_data = {
        'metadata': {
            'timestamp': timestamp,
            'model': model_name,
            'top_k': top_k,
            'total_questions': len(results),
        },
        'basic_metrics': basic_metrics,
        'ragas_metrics': ragas_scores,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=4)

    print(f"\n💾 Kết quả chi tiết: {detail_path}")
    print(f"💾 Bảng tổng hợp:   {summary_path}")
