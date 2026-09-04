"""
Reporting — Console Output & JSON Persistence
================================================
In bảng tổng hợp kết quả ra console và lưu results/summary ra JSON files.
Đặc biệt: Xử lý hiển thị thông báo Alert rõ ràng đối với Graph Database
(bỏ qua chunk retrieval metrics và ưu tiên RAGAS LLM-as-a-judge).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from .text_processing import HAS_UNDERTHESEA, safe_print

print = safe_print


class EvaluationReporter:
    """OOP Reporter để in bảng thống kê và lưu trữ kết quả đánh giá."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or settings.EVAL_RESULTS_DIR)

    def print_summary_table(self, basic_metrics: Dict[str, Any], ragas_scores: Dict[str, Any], model_name: str):
        """In bảng tổng hợp kết quả ra console với định dạng chuẩn và cảnh báo Graph mode."""
        sep = "═" * 62
        thin_sep = "─" * 62

        is_graph_mode = basic_metrics.get("is_graph_mode", False)
        top_k = basic_metrics.get("top_k", "?")

        print(f"\n{sep}")
        print(f"  [RAG EVALUATION RESULTS] - Luật Đất đai 2024")
        print(f"  Model:      {model_name}")
        print(f"  Mode:       {'Graph Database / GraphRAG' if is_graph_mode else 'Standard / Dense / Hybrid RAG'}")
        if not is_graph_mode:
            print(f"  Top-K:      {top_k}")
        print(f"{sep}\n")

        # ── 1. Retrieval Metrics ─────────────────────────────────
        if is_graph_mode:
            print(f"  {'RETRIEVAL METRICS (GRAPH DATABASE)':^58}")
            print(f"  {thin_sep}")
            print("  [ALERT] Bỏ qua các chỉ số retrieval theo văn bản thô (Recall@K,")
            print("          Precision@K, MRR, nDCG) vì Graph Database truy xuất theo")
            print("          thực thể (Entities) và cộng đồng (Communities).")
            print("  --> Đánh giá chất lượng retrieval thông qua RAGAS Context Metrics.")
            print(f"  {thin_sep}\n")
        else:
            print(f"  {'RETRIEVAL METRICS (CHUNKS)':^58}")
            print(f"  {thin_sep}")
            print(f"  {'Metric':<35} {'Value':>20}")
            print(f"  {thin_sep}")
            print(f"  {'Hit Rate':<35} {basic_metrics.get('retrieval_hit_rate', 0):.2%}")
            print(f"  {'MRR (Mean Reciprocal Rank)':<35} {basic_metrics.get('mrr', 0):.4f}")
            print(f"  {f'Recall@{top_k}':<35} {basic_metrics.get('avg_recall_at_k', 0):.4f}")
            print(f"  {f'Precision@{top_k}':<35} {basic_metrics.get('avg_precision_at_k', 0):.4f}")
            print(f"  {'nDCG':<35} {basic_metrics.get('avg_ndcg', 0):.4f}")
            print(f"  {'Avg Retrieval Latency':<35} {basic_metrics.get('avg_retrieval_latency_ms', 0):.1f} ms")
            print()

        # ── 2. Generation Metrics ────────────────────────────────
        print(f"  {'GENERATION METRICS (TEXT GROUND TRUTH)':^58}")
        print(f"  {thin_sep}")
        print(f"  {'Metric':<35} {'Value':>20}")
        print(f"  {thin_sep}")
        print(f"  {'Exact Match Rate':<35} {basic_metrics.get('exact_match_rate', 0):.2%}")
        print(f"  {'Avg F1 Score':<35} {basic_metrics.get('avg_f1', 0):.4f}")
        print(f"  {'Avg BLEU-1':<35} {basic_metrics.get('avg_bleu_1', 0):.4f}")
        print(f"  {'Avg ROUGE-L':<35} {basic_metrics.get('avg_rouge_l', 0):.4f}")
        print(f"  {'Avg Generation Latency':<35} {basic_metrics.get('avg_generation_latency_ms', 0):.1f} ms")
        print(f"  {'Avg E2E Latency':<35} {basic_metrics.get('avg_e2e_latency_ms', 0):.1f} ms")
        print()

        # ── 3. RAGAS Metrics (LLM-as-Judge) ─────────────────────
        if ragas_scores:
            print(f"  {'RAGAS METRICS (LLM-as-Judge)':^58}")
            print(f"  {thin_sep}")
            print(f"  {'Metric':<35} {'Value':>20}")
            print(f"  {thin_sep}")
            for metric, value in ragas_scores.items():
                label = metric.replace("_", " ").title()
                print(f"  {label:<35} {value:>20.4f}")
            print()
        elif is_graph_mode:
            print(f"  {'RAGAS METRICS (LLM-as-Judge)':^58}")
            print(f"  {thin_sep}")
            print("  [CHÚ Ý] Bạn đã bỏ qua RAGAS (--skip-ragas). Đối với Graph DB,")
            print("          hãy bật RAGAS để có đánh giá ngữ cảnh và chất lượng toàn diện.")
            print()

        # ── 4. Per Question Type Breakdown ──────────────────────
        per_type = basic_metrics.get("per_question_type", {})
        if per_type:
            print(f"  {'BREAKDOWN BY QUESTION TYPE':^58}")
            print(f"  {thin_sep}")
            if not is_graph_mode:
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
            else:
                print(f"  {'Type':<14} {'N':>4} {'F1':>10} {'BLEU':>10} {'ROUGE-L':>10}")
                print(f"  {thin_sep}")
                for qt, stats in per_type.items():
                    print(
                        f"  {qt:<14} {stats['count']:>4}"
                        f" {stats['avg_f1']:>10.4f}"
                        f" {stats['avg_bleu_1']:>10.4f}"
                        f" {stats['avg_rouge_l']:>10.4f}"
                    )
            print()

        print(f"{sep}\n")

    def save_results(
        self,
        results: List[Dict[str, Any]],
        basic_metrics: Dict[str, Any],
        ragas_scores: Dict[str, Any],
        eval_metadata: Dict[str, Any],
        model_name: str,
        top_k: int,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Lưu kết quả chi tiết và báo cáo tóm tắt ra các file JSON."""
        target_dir = Path(output_dir or self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clean payloads để lưu JSON an toàn
        clean_results = []
        for r in results:
            clean_item = {k: v for k, v in r.items() if k != "retrieved_payloads"}
            clean_results.append(clean_item)

        detail_path = target_dir / f"eval_results_{timestamp}.json"
        detail_data = {
            "metadata": {
                "timestamp": timestamp,
                "model": model_name,
                "top_k": top_k,
                "is_graph_mode": basic_metrics.get("is_graph_mode", False),
                "total_questions": len(results),
                "eval_source": eval_metadata,
                "tokenizer": "underthesea" if HAS_UNDERTHESEA else "simple_split",
            },
            "results": clean_results,
        }
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail_data, f, ensure_ascii=False, indent=4)

        summary_path = target_dir / f"eval_summary_{timestamp}.json"
        summary_data = {
            "metadata": {
                "timestamp": timestamp,
                "model": model_name,
                "top_k": top_k,
                "is_graph_mode": basic_metrics.get("is_graph_mode", False),
                "total_questions": len(results),
            },
            "basic_metrics": basic_metrics,
            "ragas_metrics": ragas_scores,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)

        print(f"Chi tiết kết quả lưu tại: {detail_path}")
        print(f"Báo cáo tóm tắt lưu tại:  {summary_path}")
        return detail_path, summary_path

    def save_unified_report(
        self,
        unified_output: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Lưu báo cáo chuẩn hóa đồng bộ phục vụ giao diện UI và lưu trữ dài hạn theo template eval_report_YYYYMMDD_HHMMSS.json."""
        target_dir = Path(output_dir or self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"eval_report_{timestamp}.json"
        detail_path = target_dir / report_filename
        latest_path = target_dir / "eval_latest.json"

        if "metadata" in unified_output:
            unified_output["metadata"]["report_filename"] = report_filename
            unified_output["metadata"]["saved_at"] = datetime.now().isoformat()

        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(unified_output, f, ensure_ascii=False, indent=4)

        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(unified_output, f, ensure_ascii=False, indent=4)

        print(f"Báo cáo chi tiết đã lưu tại:  {detail_path}")
        print(f"Bản đồng bộ mới nhất đã lưu: {latest_path}")
        return detail_path, latest_path


# Backward compatibility
def print_summary_table(basic_metrics: dict, ragas_scores: dict, model_name: str):
    reporter = EvaluationReporter()
    reporter.print_summary_table(basic_metrics, ragas_scores, model_name)


def save_results(results, basic_metrics, ragas_scores, eval_metadata, model_name, top_k, output_dir=None):
    reporter = EvaluationReporter(output_dir)
    clean_results = [{k: v for k, v in r.items() if k != "retrieved_payloads"} for r in results]
    unified = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "top_k": top_k,
            "eval_source": eval_metadata,
        },
        "summary_metrics": {
            "retrieval": basic_metrics,
            "ragas": ragas_scores,
        },
        "detailed_results": clean_results,
    }
    return reporter.save_unified_report(unified, output_dir=output_dir)
