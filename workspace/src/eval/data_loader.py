"""
Data Loader — Eval Dataset
============================
Load và validate eval dataset từ JSON file.
"""

import json
from pathlib import Path
from src.eval.text_processing import safe_print

print = safe_print


def load_eval_dataset(eval_file: Path, limit: int = None) -> tuple[dict, list[dict]]:
    """Load eval dataset từ JSON file.

    Args:
        eval_file: Đường dẫn đến file eval JSON.
        limit: Giới hạn số câu hỏi (None = tất cả).

    Returns:
        (metadata, data_list) — metadata dict và list câu hỏi (có thể đã truncate).
    """
    with open(eval_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    metadata = raw.get('metadata', {})
    data = raw.get('data', [])

    if limit and limit > 0:
        data = data[:limit]

    print(f"Loaded {len(data)} questions from {eval_file.name}")
    print(f"   Law: {metadata.get('law_name', 'N/A')} | Version: {metadata.get('version', 'N/A')}")
    return metadata, data
