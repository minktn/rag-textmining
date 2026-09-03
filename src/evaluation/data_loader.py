"""
Data Loader — Eval Dataset
============================
Load, validate và sample eval dataset từ JSON file (Luật Đất đai 2024).
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from .text_processing import safe_print


class EvalDataLoader:
    """OOP Loader cho evaluation datasets."""

    def __init__(self, default_file: Optional[Path] = None):
        self.default_file = default_file or (settings.EVAL_DATA_DIR / "eval_landlaw_2024.json")

    def load(
        self,
        eval_file: Optional[Path] = None,
        limit: Optional[int] = None,
        random_sample: bool = False,
        seed: int = 42,
        question_types: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Load và lọc eval dataset từ JSON file."""
        target_file = Path(eval_file or self.default_file)
        if not target_file.exists():
            raise FileNotFoundError(f"Không tìm thấy file eval dataset tại: {target_file}")

        with open(target_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        metadata = raw.get("metadata", {})
        data = raw.get("data", [])

        # Lọc theo question_types nếu có
        if question_types:
            q_set = set(question_types)
            data = [d for d in data if d.get("question_type") in q_set]

        # Sampling hoặc Truncate
        if random_sample and limit and limit < len(data):
            rng = random.Random(seed)
            data = rng.sample(data, limit)
        elif limit and limit > 0:
            data = data[:limit]

        safe_print(
            f"Loaded {len(data)} questions from {target_file.name} "
            f"| Law: {metadata.get('law_name', 'N/A')} "
            f"| Version: {metadata.get('version', 'N/A')}"
        )
        return metadata, data


# Backward-compatible function
def load_eval_dataset(eval_file: Path, limit: Optional[int] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Hàm helper tương thích ngược."""
    loader = EvalDataLoader()
    return loader.load(eval_file=eval_file, limit=limit)
