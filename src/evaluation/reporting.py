"""
Reporting — Console Output & JSON Persistence
================================================
In bảng tổng hợp kết quả ra console và lưu results/summary ra JSON files.
Đặc biệt: Xử lý hiển thị thông báo Alert rõ ràng đối với Graph Database
(bỏ qua chunk retrieval metrics và ưu tiên RAGAS LLM-as-a-judge).
"""

import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import settings
from .text_processing import HAS_UNDERTHESEA, safe_print

print = safe_print


class EvaluationReporter:
    """OOP Reporter để in bảng thống kê, streaming append kết quả theo case và tự động khôi phục phiên chạy (Auto-Resume)."""

    CRITICAL_CONFIG_KEYS = [
        "eval_file",
        "retriever_mode",
        "advanced_method",
        "preprocessing",
        "postprocessing",
        "llm_service",
        "llm_model",
        "sub_llm_service",
        "sub_llm_model",
        "top_k",
        "collection_name",
    ]

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or settings.EVAL_RESULTS_DIR)
        self._lock = threading.Lock()

    @staticmethod
    def build_pipeline_signature(config: Dict[str, Any]) -> str:
        """
        Sinh chuỗi signature duy nhất phản ánh cấu hình pipeline đánh giá.
        Quy tắc:
          1. retriever_mode (base | contriever | graph)
          2. advanced_method (nếu có) HOẶC [preprocessing...] + [postprocessing...]
          3. llm_service
          4. sub_llm_service
          5. ragas_service (hoặc 'skip_ragas')
        Ví dụ:
          - base_hyde_crag_nvidia_nvidia_google
          - base_rag_fusion_nvidia_nvidia_google
          - base_filter_rerank_google_google_nvidia
          - graph_filter_rerank_nvidia_nvidia_google
        """
        parts: List[str] = []

        # 1. Retriever mode
        mode = (config.get("retriever_mode") or "base").lower().strip()
        parts.append(mode)

        # 2. Methods: Advanced HOẶC Preprocessing & Postprocessing
        method_added = False
        if config.get("advanced_method"):
            parts.append(str(config.get("advanced_method")).lower().strip())
            method_added = True
        else:
            pre = config.get("preprocessing") or []
            if isinstance(pre, list):
                for p in pre:
                    if p:
                        parts.append(str(p).lower().strip())
                        method_added = True
            elif isinstance(pre, str) and pre:
                parts.append(pre.lower().strip())
                method_added = True

            post = config.get("postprocessing") or []
            if isinstance(post, list):
                for p in post:
                    if p:
                        parts.append(str(p).lower().strip())
                        method_added = True
            elif isinstance(post, str) and post:
                parts.append(post.lower().strip())
                method_added = True


        # 3. LLM Service
        llm = (config.get("llm_service") or getattr(settings, "LLM_SERVICE", "nvidia")).lower().strip()
        parts.append(llm)

        # 4. Sub-LLM Service
        sub_llm = (config.get("sub_llm_service") or getattr(settings, "SUB_LLM_SERVICE", None) or llm).lower().strip()
        parts.append(sub_llm)

        # 5. RAGAS Service
        if config.get("skip_ragas"):
            parts.append("skip_ragas")
        else:
            ragas = (config.get("ragas_service") or getattr(settings, "RAGAS_SERVICE", "google") or "google").lower().strip()
            parts.append(ragas)

        clean_parts = [p.replace("-", "_").replace(" ", "_") for p in parts if p]
        return "_".join(clean_parts)

    def get_latest_filepath(self, config: Dict[str, Any], target_dir: Optional[Path] = None) -> Path:
        td = Path(target_dir or self.output_dir)
        sig = self.build_pipeline_signature(config)
        return td / f"eval_latest_{sig}.json"

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
            metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            for m in metric_names:
                if m in ragas_scores:
                    val = ragas_scores[m]
                    label = m.replace("_", " ").title()
                    val_str = f"{val:.4f}" if val is not None else "N/A"
                    print(f"  {label:<35} {val_str:>20}")
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

    # ─────────────────────────────────────────────────────────────
    # Auto-Resume & Session Matching
    # ─────────────────────────────────────────────────────────────

    def find_resumable_session(
        self,
        current_config: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra file kết quả gần nhất xem có trùng các trường cấu hình quan trọng hay không.
        Nếu trùng khớp, trả về session để tiếp tục append các câu hỏi chưa chạy.
        """
        target_dir = Path(output_dir or self.output_dir)
        candidate_files: List[Path] = []

        # Ưu tiên kiểm tra file latest của chính cấu hình này
        specific_latest = self.get_latest_filepath(current_config, target_dir)
        if specific_latest.exists():
            candidate_files.append(specific_latest)

        # Fallback các file eval_latest_*.json khác và eval_latest.json
        for lf in sorted(target_dir.glob("eval_latest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if lf not in candidate_files:
                candidate_files.append(lf)

        latest_path = target_dir / "eval_latest.json"
        if latest_path.exists() and latest_path not in candidate_files:
            candidate_files.append(latest_path)

        # Tìm các file eval_report_*.json gần nhất
        report_files = sorted(
            target_dir.glob("eval_report_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for rf in report_files[:10]:
            if rf not in candidate_files:
                candidate_files.append(rf)

        for filepath in candidate_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                meta = data.get("metadata") or {}
                prev_config = meta.get("configuration") or {}
                if not prev_config:
                    continue

                # So sánh các trường cấu hình quan trọng
                is_match = True
                for key in self.CRITICAL_CONFIG_KEYS:
                    curr_val = current_config.get(key)
                    prev_val = prev_config.get(key)

                    # Chuẩn hóa so sánh cho danh sách (ví dụ: preprocessing/postprocessing)
                    if isinstance(curr_val, list) and isinstance(prev_val, list):
                        if sorted(curr_val) != sorted(prev_val):
                            is_match = False
                            break
                    elif curr_val != prev_val:
                        # Cho phép linh hoạt nếu một bên là None và bên kia là rỗng/mặc định
                        if curr_val in (None, "") and prev_val in (None, ""):
                            continue
                        is_match = False
                        break

                if not is_match:
                    continue

                # Kiểm tra sampling ngẫu nhiên nếu có
                if current_config.get("random_sample"):
                    if not prev_config.get("random_sample") or current_config.get("seed") != prev_config.get("seed"):
                        continue

                detailed = data.get("detailed_results") or []
                completed_ids: Set[str] = {
                    item["id"] for item in detailed if isinstance(item, dict) and item.get("id")
                }

                # Xác định file báo cáo thực sự
                report_filename = meta.get("report_filename")
                actual_report_path = (target_dir / report_filename) if report_filename else filepath
                if not actual_report_path.exists():
                    actual_report_path = filepath

                return {
                    "report_path": actual_report_path,
                    "report_data": data,
                    "completed_ids": completed_ids,
                    "completed_results": detailed,
                    "matched_file": filepath.name,
                }

            except Exception as e:
                logger_msg = f"[Reporter] Bỏ qua file '{filepath.name}' khi check resume: {e}"
                continue

        return None

    # ─────────────────────────────────────────────────────────────
    # Streaming Append & Lifecycle Management
    # ─────────────────────────────────────────────────────────────

    def init_streaming_session(
        self,
        metadata: Dict[str, Any],
        total_questions: int,
        output_dir: Optional[Path] = None,
        existing_report_path: Optional[Path] = None,
    ) -> Path:
        """
        Khởi tạo file JSON ngay từ đầu phiên để sẵn sàng lưu theo phương pháp append.
        """
        target_dir = Path(output_dir or self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if existing_report_path and existing_report_path.exists():
            report_path = existing_report_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"eval_report_{timestamp}.json"
            report_path = target_dir / report_filename
            metadata["report_filename"] = report_filename

        metadata["status"] = "in_progress"
        metadata["total_questions"] = total_questions
        metadata["created_at"] = metadata.get("created_at") or datetime.now().isoformat()
        metadata["last_updated_at"] = datetime.now().isoformat()

        initial_data = {
            "metadata": metadata,
            "summary_metrics": {
                "status": "in_progress",
                "completed_count": 0,
                "total_count": total_questions,
                "is_graph_mode": metadata.get("is_graph_mode", False),
                "retrieval": {},
                "generation": {},
                "ragas": {},
            },
            "detailed_results": [],
        }

        config = metadata.get("configuration") or {}
        sig = self.build_pipeline_signature(config)
        metadata["pipeline_signature"] = sig
        latest_path = self.get_latest_filepath(config, target_dir)
        metadata["latest_filename"] = latest_path.name

        with self._lock:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=4)

            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=4)

        return report_path

    def append_case_result(
        self,
        report_path: Path,
        case_result: Dict[str, Any],
        total_questions: int,
        output_dir: Optional[Path] = None,
    ):
        """
        Ghi nhận kết quả của 1 case vừa xử lý xong vào JSON theo cơ chế append an toàn.
        """
        target_dir = Path(output_dir or self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Làm sạch payload không cần thiết trước khi lưu
        clean_item = {k: v for k, v in case_result.items() if k != "retrieved_payloads"}

        with self._lock:
            # Đọc dữ liệu hiện tại
            current_data = {}
            if report_path.exists():
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        current_data = json.load(f)
                except Exception:
                    current_data = {}

            if "detailed_results" not in current_data:
                current_data["detailed_results"] = []

            # Tránh ghi đè trùng case id nếu đã tồn tại
            existing_indices = [
                idx for idx, item in enumerate(current_data["detailed_results"])
                if item.get("id") == clean_item.get("id")
            ]
            if existing_indices:
                current_data["detailed_results"][existing_indices[0]] = clean_item
            else:
                current_data["detailed_results"].append(clean_item)

            completed = len(current_data["detailed_results"])

            if "metadata" not in current_data:
                current_data["metadata"] = {}
            current_data["metadata"]["completed_questions"] = completed
            current_data["metadata"]["total_questions"] = total_questions
            current_data["metadata"]["last_updated_at"] = datetime.now().isoformat()
            current_data["metadata"]["status"] = "in_progress"

            if "summary_metrics" not in current_data:
                current_data["summary_metrics"] = {}
            current_data["summary_metrics"]["status"] = "in_progress"
            current_data["summary_metrics"]["completed_count"] = completed
            current_data["summary_metrics"]["total_count"] = total_questions
            current_data["summary_metrics"]["is_graph_mode"] = current_data.get("metadata", {}).get("is_graph_mode", False)

            # Tính toán và tổng hợp kết quả trung bình lũy kế ở đầu JSON ngay khi có case mới
            is_graph = current_data["summary_metrics"]["is_graph_mode"]
            summary = self.compute_summary_from_detailed(current_data["detailed_results"], is_graph)
            current_data["summary_metrics"].update(summary)

            # Ghi đồng thời ra report_path và eval_latest_{signature}.json
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)

            config = current_data.get("metadata", {}).get("configuration") or {}
            latest_path = self.get_latest_filepath(config, target_dir)
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)

    def update_ragas_batch(
        self,
        report_path: Path,
        batch_results: List[Dict[str, Any]],
        current_ragas_summary: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ):
        """
        Cập nhật kết quả RAGAS của từng batch vào file JSON ngay lập tức (Bảo lưu checkpoint).
        Đảm bảo các case đã chấm RAGAS thành công được bảo lưu bền vững trên đĩa,
        nếu tiến trình bị gián đoạn thì lần chạy tiếp theo không phải chấm lại.
        """
        target_dir = Path(output_dir or self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        metric_keys = [
            "ragas_faithfulness",
            "ragas_answer_relevancy",
            "ragas_context_precision",
            "ragas_context_recall",
        ]

        with self._lock:
            current_data = {}
            if report_path.exists():
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        current_data = json.load(f)
                except Exception:
                    current_data = {}

            if "detailed_results" not in current_data:
                current_data["detailed_results"] = []

            item_map = {
                item.get("id"): item
                for item in current_data["detailed_results"]
                if isinstance(item, dict) and item.get("id")
            }

            for b_item in batch_results:
                qid = b_item.get("id")
                if qid and qid in item_map:
                    target_item = item_map[qid]
                    for mk in metric_keys:
                        if mk in b_item:
                            target_item[mk] = b_item[mk]
                elif qid:
                    clean_item = {k: v for k, v in b_item.items() if k != "retrieved_payloads"}
                    current_data["detailed_results"].append(clean_item)
                    item_map[qid] = clean_item

            if "metadata" not in current_data:
                current_data["metadata"] = {}
            current_data["metadata"]["last_updated_at"] = datetime.now().isoformat()

            if "summary_metrics" not in current_data:
                current_data["summary_metrics"] = {}

            # Cập nhật tổng hợp kết quả trung bình đầy đủ ở đầu JSON
            is_graph = current_data.get("metadata", {}).get("is_graph_mode", False)
            summary = self.compute_summary_from_detailed(current_data["detailed_results"], is_graph)
            current_data["summary_metrics"].update(summary)
            if current_ragas_summary:
                current_data["summary_metrics"]["ragas"] = current_ragas_summary

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)

            config = current_data.get("metadata", {}).get("configuration") or {}
            latest_path = self.get_latest_filepath(config, target_dir)
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def compute_summary_from_detailed(
        detailed_results: List[Dict[str, Any]],
        is_graph_mode: bool = False,
    ) -> Dict[str, Any]:
        """Tổng hợp macro-average cho toàn bộ các metric từ detailed_results."""
        total = len(detailed_results)
        if total == 0:
            return {
                "retrieval": {},
                "generation": {},
                "ragas": {},
                "per_question_type": {},
            }

        def _avg(vals):
            valid = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return round(sum(valid) / len(valid), 4) if valid else 0.0

        # Retrieval metrics
        hit_vals = [
            1.0 if r.get("retrieval_hit") else 0.0
            for r in detailed_results
            if "retrieval_hit" in r and r.get("retrieval_hit") is not None
        ]
        mrr_vals = [r.get("mrr") for r in detailed_results if r.get("mrr") is not None]
        rcl_vals = [r.get("recall_at_k") for r in detailed_results if r.get("recall_at_k") is not None]
        prc_vals = [r.get("precision_at_k") for r in detailed_results if r.get("precision_at_k") is not None]
        ndcg_vals = [r.get("ndcg") for r in detailed_results if r.get("ndcg") is not None]
        ret_lat_vals = [r.get("retrieval_latency_ms") for r in detailed_results if r.get("retrieval_latency_ms") is not None]

        retrieval_summary = {
            "hit_rate": round(sum(hit_vals) / len(hit_vals), 4) if hit_vals else (None if is_graph_mode else 0.0),
            "mrr": _avg(mrr_vals) if not is_graph_mode else None,
            "recall_at_k": _avg(rcl_vals) if not is_graph_mode else None,
            "precision_at_k": _avg(prc_vals) if not is_graph_mode else None,
            "ndcg": _avg(ndcg_vals) if not is_graph_mode else None,
            "latency_ms": _avg(ret_lat_vals),
            "status": "N/A (Graph Mode)" if is_graph_mode else "OK",
        }

        # Generation metrics
        em_vals = [
            1.0 if r.get("exact_match") else 0.0
            for r in detailed_results
            if "exact_match" in r and r.get("exact_match") is not None
        ]
        f1_vals = [r.get("f1_score") for r in detailed_results if r.get("f1_score") is not None]
        bleu_vals = [r.get("bleu_1") for r in detailed_results if r.get("bleu_1") is not None]
        rouge_vals = [r.get("rouge_l") for r in detailed_results if r.get("rouge_l") is not None]
        gen_lat_vals = [r.get("generation_latency_ms") for r in detailed_results if r.get("generation_latency_ms") is not None]
        e2e_lat_vals = [r.get("e2e_latency_ms") for r in detailed_results if r.get("e2e_latency_ms") is not None]

        generation_summary = {
            "exact_match_rate": round(sum(em_vals) / len(em_vals), 4) if em_vals else 0.0,
            "avg_f1": _avg(f1_vals),
            "avg_bleu_1": _avg(bleu_vals),
            "avg_rouge_l": _avg(rouge_vals),
            "latency_ms": _avg(gen_lat_vals),
            "e2e_latency_ms": _avg(e2e_lat_vals),
        }

        # RAGAS metrics
        ragas_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        ragas_summary = {}
        for m in ragas_names:
            k = f"ragas_{m}"
            vals = [
                r.get(k)
                for r in detailed_results
                if r.get(k) is not None and not (isinstance(r.get(k), float) and math.isnan(r.get(k)))
            ]
            ragas_summary[m] = round(sum(vals) / len(vals), 4) if vals else None

        # Per question type
        per_type = {}
        for r in detailed_results:
            qt = r.get("question_type") or "unknown"
            if qt not in per_type:
                per_type[qt] = {
                    "count": 0,
                    "hit_count": 0,
                    "f1_sum": 0.0,
                    "bleu_sum": 0.0,
                    "rouge_sum": 0.0,
                    "recall_sum": 0.0,
                }
            pt = per_type[qt]
            pt["count"] += 1
            if r.get("retrieval_hit"):
                pt["hit_count"] += 1
            pt["f1_sum"] += (r.get("f1_score") or 0.0)
            pt["bleu_sum"] += (r.get("bleu_1") or 0.0)
            pt["rouge_sum"] += (r.get("rouge_l") or 0.0)
            pt["recall_sum"] += (r.get("recall_at_k") or 0.0)

        per_type_summary = {}
        for qt, pt in per_type.items():
            c = pt["count"]
            per_type_summary[qt] = {
                "count": c,
                "hit_rate": round(pt["hit_count"] / c, 4) if c else 0.0,
                "avg_recall_at_k": round(pt["recall_sum"] / c, 4) if c else 0.0,
                "avg_f1": round(pt["f1_sum"] / c, 4) if c else 0.0,
                "avg_bleu_1": round(pt["bleu_sum"] / c, 4) if c else 0.0,
                "avg_rouge_l": round(pt["rouge_sum"] / c, 4) if c else 0.0,
            }

        return {
            "retrieval": retrieval_summary,
            "generation": generation_summary,
            "ragas": ragas_summary,
            "per_question_type": per_type_summary,
        }

    def finalize_unified_report(
        self,
        unified_output: Dict[str, Any],
        report_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """
        Hoàn tất báo cáo đánh giá (đánh dấu status='completed', cập nhật metrics và ghi file cuối cùng).
        """
        target_dir = Path(output_dir or self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"eval_report_{timestamp}.json"
            report_path = target_dir / report_filename
        else:
            report_filename = report_path.name

        config = unified_output.get("metadata", {}).get("configuration") or {}
        sig = self.build_pipeline_signature(config)
        latest_path = self.get_latest_filepath(config, target_dir)

        if "metadata" in unified_output:
            unified_output["metadata"]["report_filename"] = report_filename
            unified_output["metadata"]["pipeline_signature"] = sig
            unified_output["metadata"]["latest_filename"] = latest_path.name
            unified_output["metadata"]["saved_at"] = datetime.now().isoformat()
            unified_output["metadata"]["completed_at"] = datetime.now().isoformat()
            unified_output["metadata"]["status"] = "completed"

        with self._lock:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(unified_output, f, ensure_ascii=False, indent=4)

            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(unified_output, f, ensure_ascii=False, indent=4)

        print(f"Báo cáo chi tiết đã lưu tại:  {report_path}")
        print(f"Bản đồng bộ mới nhất đã lưu: {latest_path}")
        return report_path, latest_path

    # Backward compatibility
    def save_unified_report(
        self,
        unified_output: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        return self.finalize_unified_report(unified_output, output_dir=output_dir)

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
        """Lưu kết quả chi tiết và báo cáo tóm tắt ra các file JSON (tương thích ngược)."""
        target_dir = Path(output_dir or self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_results = [{k: v for k, v in r.items() if k != "retrieved_payloads"} for r in results]

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
    return reporter.finalize_unified_report(unified, output_dir=output_dir)
