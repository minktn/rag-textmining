"""
CRAG Utility Functions
======================
Hàm phụ trợ cho việc tiền xử lý văn bản pháp lý, trích xuất dải tri thức (knowledge strips),
chọn lọc câu liên quan (select_relevants), và phân loại hành động CRAG.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from .evaluator import BamiBERTRetrievalEvaluator


def extract_strips_from_passage(
	passage: str,
	mode: str = "excerption",
	window_length: int = 50,
	min_strip_chars: int = 20,
) -> List[str]:
	"""Phân rã một đoạn văn bản pháp luật thành các dải tri thức (strips).

	Parameters
	----------
	passage : str
		Đoạn văn bản pháp luật gốc.
	mode : str
		- 'selection': Giữ nguyên toàn bộ passage làm 1 strip.
		- 'excerption': Tách câu theo dấu chấm/chấm phẩy/xuống dòng của văn bản luật.
		- 'fixed_num': Tách theo số lượng từ cố định (sliding window).
	"""
	if not passage or not passage.strip():
		return []

	text = passage.strip()

	if mode == "selection":
		return [text]

	elif mode == "fixed_num":
		words = text.split()
		strips = []
		buf = []
		for w in words:
			buf.append(w)
			if len(buf) == window_length:
				strips.append(" ".join(buf))
				buf = []
		if buf:
			if len(buf) < 10 and strips:
				strips[-1] += " " + " ".join(buf)
			else:
				strips.append(" ".join(buf))
		return strips

	else:  # mode == 'excerption' (mặc định cho văn bản luật)
		raw_sentences = re.split(r"(?<=[.!?;\n])\s+", text)
		strips = [s.strip() for s in raw_sentences if len(s.strip()) >= min_strip_chars]
		return strips if strips else [text]


def select_relevants(
	strips: List[str],
	query: str,
	evaluator: BamiBERTRetrievalEvaluator,
	top_n: int = 3,
	min_threshold: float = 0.35,
) -> Tuple[str, List[int], List[float]]:
	"""Đánh giá và chọn lọc các dải tri thức (strips) có độ liên quan cao nhất với câu hỏi.

	Returns
	-------
	Tuple[str, List[int], List[float]]
		(joined_selected_text, selected_indices, selected_scores)
	"""
	if not strips:
		return "", [], []

	scores = evaluator.evaluate_batch(query, strips)
	indexed_scores = list(enumerate(scores))

	# Lọc theo ngưỡng tối thiểu
	valid_indexed = [item for item in indexed_scores if item[1] >= min_threshold]

	# Nếu không có câu nào vượt ngưỡng, lấy câu cao điểm nhất
	if not valid_indexed:
		valid_indexed = [max(indexed_scores, key=lambda x: x[1])]

	# Sắp xếp theo điểm giảm dần và lấy top_n
	valid_indexed.sort(key=lambda x: x[1], reverse=True)
	top_items = valid_indexed[:top_n]

	# Sắp xếp lại theo thứ tự xuất hiện ban đầu để giữ mạch văn bản
	top_items.sort(key=lambda x: x[0])

	selected_indices = [item[0] for item in top_items]
	selected_scores = [item[1] for item in top_items]
	selected_strips = [strips[i] for i in selected_indices]

	return " ".join(selected_strips), selected_indices, selected_scores
