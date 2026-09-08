"""
Offline RAGAS Evaluation Script
===============================
Chạy đánh giá RAGAS (LLM-as-a-judge) độc lập từ file JSON báo cáo đã có,
không cần chạy lại quá trình Retrieval và Generation.

Cách chạy:
    # 1. Đánh giá file kết quả mới nhất với NVIDIA NIM (mặc định)
    python Scripts/evaluate_ragas_offline.py

    # 2. Chỉ định file kết quả cụ thể
    python Scripts/evaluate_ragas_offline.py --report db/results/eval_report_20260907_193000.json

    # 3. Sử dụng Google (Gemini 1.5 Flash) để chấm điểm siêu nhanh
    python Scripts/evaluate_ragas_offline.py --service google --model gemini-1.5-flash --batch-size 10

    # 4. Tùy chỉnh batch size và số worker
    python Scripts/evaluate_ragas_offline.py --batch-size 10 --max-workers 2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
	sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
	sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.evaluation.ragas_metrics import RagasJudge
from src.evaluation.reporting import EvaluationReporter
from src.evaluation.text_processing import safe_print

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_latest_report(results_dir: Path) -> Optional[Path]:
	"""Tìm file eval_report_*.json mới nhất trong thư mục results."""
	if not results_dir.exists():
		return None
	files = [
		p for p in results_dir.glob("eval_report_*.json")
		if not p.name.endswith(".tmp")
	]
	if not files:
		# Thử tìm eval_latest.json
		latest = results_dir / "eval_latest.json"
		return latest if latest.exists() else None
	files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	return files[0]


def parse_args():
	parser = argparse.ArgumentParser(
		description="Đánh giá RAGAS độc lập từ file báo cáo JSON đã có."
	)
	parser.add_argument(
		"--report",
		type=str,
		default=None,
		help="Đường dẫn đến file eval_report_*.json cần chấm điểm (mặc định: file mới nhất trong db/results/)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Đường dẫn file lưu kết quả (mặc định: cập nhật đè trực tiếp vào file report gốc)",
	)
	parser.add_argument(
		"--service",
		type=str,
		default=None,
		choices=["nvidia", "google", "groq"],
		help="Dịch vụ LLM Judge (mặc định: theo settings.RAGAS_SERVICE hoặc 'nvidia')",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=None,
		help="Tên model LLM Judge (ví dụ: 'nvidia/nemotron-3-ultra-550b-a55b' hoặc 'gemini-1.5-flash')",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=10,
		help="Kích thước batch khi gửi tới RAGAS (mặc định: 10 câu/batch)",
	)
	parser.add_argument(
		"--max-workers",
		type=int,
		default=2,
		help="Số luồng gửi đồng thời (mặc định: 2 workers)",
	)
	parser.add_argument(
		"--only-missing",
		action="store_true",
		help="Chỉ chấm điểm cho các câu hỏi còn thiếu RAGAS (chưa có điểm)",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	# 1. Xác định file báo cáo đầu vào
	if args.report:
		report_path = Path(args.report).resolve()
	else:
		report_path = find_latest_report(settings.EVAL_RESULTS_DIR)

	if not report_path or not report_path.exists():
		safe_print(f"[Lỗi] Không tìm thấy file báo cáo kết quả tại: {report_path}")
		safe_print(f"Vui lòng chỉ định đường dẫn cụ thể bằng cờ --report <đường_dẫn_file.json>")
		sys.exit(1)

	safe_print("=" * 65)
	safe_print("  [OFFLINE RAGAS EVALUATION] - Đánh giá RAGAS từ file đã lưu")
	safe_print("=" * 65)
	safe_print(f"  File báo cáo:   {report_path.name}")
	safe_print(f"  Đường dẫn:      {report_path}")

	# 2. Đọc dữ liệu từ file JSON
	with open(report_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	detailed_results: List[Dict[str, Any]] = data.get("detailed_results", [])
	if not detailed_results:
		safe_print("[Lỗi] File báo cáo không chứa danh sách 'detailed_results'!")
		sys.exit(1)

	total_items = len(detailed_results)
	safe_print(f"  Tổng số câu hỏi: {total_items} câu")

	# 3. Lọc danh sách câu cần chấm điểm nếu bật --only-missing
	metric_keys = ["ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision", "ragas_context_recall"]
	if args.only_missing:
		items_to_eval = [
			r for r in detailed_results
			if any(r.get(m) is None for m in metric_keys)
		]
		safe_print(f"  Chế độ --only-missing: Cần chấm bổ sung {len(items_to_eval)}/{total_items} câu.")
	else:
		items_to_eval = detailed_results

	if not items_to_eval:
		safe_print("  ✓ Tất cả câu hỏi đã có đầy đủ điểm RAGAS! Không cần chấm thêm.")
		return

	# 4. Khởi tạo RagasJudge
	judge_service = args.service or getattr(settings, "RAGAS_SERVICE", "nvidia")
	judge = RagasJudge(
		service=judge_service,
		model_name=args.model,
		batch_size=args.batch_size,
		max_workers=args.max_workers,
	)

	safe_print(f"  Dịch vụ Judge:  {judge.service.upper()} ({judge.model_name})")
	safe_print(f"  Batch size:     {args.batch_size} câu/batch")
	safe_print(f"  Số luồng chạy:  {args.max_workers} workers")
	safe_print("=" * 65)

	# 5. Thực thi đánh giá RAGAS
	safe_print(f"\nBắt đầu chấm điểm RAGAS cho {len(items_to_eval)} câu hỏi...")
	new_ragas_scores = judge.evaluate(
		items_to_eval,
		batch_size=args.batch_size,
		max_workers=args.max_workers,
	)

	# 6. Tính toán lại điểm số trung bình toàn cục (macro-average) trên toàn bộ 205 câu
	macro_ragas: Dict[str, float] = {}
	metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
	for m in metric_names:
		key = f"ragas_{m}"
		vals = [r[key] for r in detailed_results if r.get(key) is not None]
		if vals:
			macro_ragas[m] = round(sum(vals) / len(vals), 4)

	# 7. Cập nhật lại cấu trúc JSON
	if "summary_metrics" not in data:
		data["summary_metrics"] = {}
	data["summary_metrics"]["ragas"] = macro_ragas

	if "metadata" in data and "configuration" in data["metadata"]:
		data["metadata"]["configuration"]["ragas_service"] = judge.service
		data["metadata"]["configuration"]["ragas_model"] = judge.model_name
		data["metadata"]["configuration"]["ragas_batch_size"] = args.batch_size

	# 8. Lưu file
	output_path = Path(args.output).resolve() if args.output else report_path
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

	safe_print(f"\n✓ Đã cập nhật điểm RAGAS thành công vào: {output_path}")

	# 9. In bảng tổng kết kết quả đẹp mắt
	reporter = EvaluationReporter()
	retrieval_stats = data.get("summary_metrics", {}).get("retrieval", {})
	generation_stats = data.get("summary_metrics", {}).get("generation", {})

	basic_metrics = {
		"retrieval_hit_rate": retrieval_stats.get("hit_rate"),
		"mrr": retrieval_stats.get("mrr"),
		"avg_recall_at_k": retrieval_stats.get("recall_at_k"),
		"avg_precision_at_k": retrieval_stats.get("precision_at_k"),
		"avg_ndcg": retrieval_stats.get("ndcg"),
		"avg_f1": generation_stats.get("avg_f1"),
		"avg_bleu_1": generation_stats.get("avg_bleu_1"),
		"avg_rouge_l": generation_stats.get("avg_rouge_l"),
		"avg_retrieval_latency_ms": retrieval_stats.get("latency_ms"),
		"avg_generation_latency_ms": generation_stats.get("latency_ms"),
		"avg_e2e_latency_ms": generation_stats.get("e2e_latency_ms"),
	}

	model_label = (
		data.get("metadata", {}).get("configuration", {}).get("llm_model")
		or "RAG Pipeline"
	)
	reporter.print_summary_table(basic_metrics, macro_ragas, model_label)


if __name__ == "__main__":
	main()

