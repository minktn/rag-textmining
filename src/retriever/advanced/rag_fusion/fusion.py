from typing import Any, Dict, List, Optional, Union


def get_doc_id(doc: Any, key_field: str = "id") -> str:
	"""Trích xuất ID hoặc identifier độc nhất của document."""
	if isinstance(doc, dict):
		if key_field in doc:
			return str(doc[key_field])
		if "payload" in doc and isinstance(doc["payload"], dict) and key_field in doc["payload"]:
			return str(doc["payload"][key_field])
		if "content" in doc:
			# Dùng hash nội dung nếu không có id
			return str(hash(doc["content"]))
		return str(doc)
	elif hasattr(doc, key_field):
		return str(getattr(doc, key_field))
	elif hasattr(doc, "payload") and isinstance(doc.payload, dict) and key_field in doc.payload:
		return str(doc.payload[key_field])
	else:
		return str(hash(str(doc)))


def to_dict_doc(doc: Any) -> Dict[str, Any]:
	"""Chuyển đổi các đối tượng document/point thành dictionary tiêu chuẩn."""
	if isinstance(doc, dict):
		return dict(doc)
	elif hasattr(doc, "model_dump"):
		return doc.model_dump()
	elif hasattr(doc, "__dict__"):
		return dict(doc.__dict__)
	else:
		return {"content": str(doc)}


def reciprocal_rank_fusion(
	search_results_dict: Dict[str, List[Any]],
	k: int = 60,
	query_weights: Optional[Dict[str, float]] = None,
	key_field: str = "id",
) -> List[Dict[str, Any]]:
	"""Thực hiện Reciprocal Rank Fusion (RRF) gộp các danh sách kết quả tìm kiếm.

	Parameters
	----------
	search_results_dict : Dict[str, List[Any]]
		Dictionary mapping giữa câu hỏi (query) và danh sách các tài liệu trả về.
	k : int, default=60
		Hằng số làm mượt RRF.
	query_weights : Optional[Dict[str, float]], default=None
		Trọng số cho từng câu hỏi (ví dụ câu gốc có trọng số 3.0, câu phụ trọng số 1.0).
	key_field : str, default="id"
		Tên trường định danh của tài liệu.

	Returns
	-------
	List[Dict[str, Any]]
		Danh sách các tài liệu đã được hợp nhất và xếp hạng lại theo điểm rrf_score giảm dần.
	"""
	fused_scores: Dict[str, float] = {}
	doc_store: Dict[str, Dict[str, Any]] = {}

	for query, doc_list in search_results_dict.items():
		weight = query_weights.get(query, 1.0) if query_weights else 1.0

		for rank, doc in enumerate(doc_list):
			doc_id = get_doc_id(doc, key_field=key_field)

			if doc_id not in doc_store:
				doc_dict = to_dict_doc(doc)
				doc_dict["id"] = doc_id
				doc_store[doc_id] = doc_dict

			if doc_id not in fused_scores:
				fused_scores[doc_id] = 0.0

			# Công thức RRF: weight * (1 / (rank + k))
			fused_scores[doc_id] += weight * (1.0 / (rank + k))

	# Gán điểm rrf_score và sắp xếp lại
	reranked_docs = []
	for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
		doc = doc_store[doc_id]
		doc["rrf_score"] = score
		reranked_docs.append(doc)

	return reranked_docs
