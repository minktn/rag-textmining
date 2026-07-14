from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from src.common import LegalMetadataProcessor

if TYPE_CHECKING:
	from qdrant_client.models import Filter

	from src.data_pipeline import DenseEmbedder
	from src.database import DBManager


class Retriever:
	"""Retrieval pipeline for legal RAG queries."""

	DEFAULT_COLLECTION_NAME = "landlaw"
	DEFAULT_DENSE_MODEL = "keepitreal/vietnamese-sbert"
	DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
	DEFAULT_DENSE_CANDIDATE_LIMIT = 20
	DEFAULT_RERANK_LIMIT = 5
	DEFAULT_EXPANSION_ARTICLE_LIMIT = 20
	DEFAULT_MAX_CONTEXT_CHARS = 12000

	FILTER_FIELDS = ("article_no", "chapter_no", "section_no")

	def __init__(
		self,
		db_manager: DBManager | None = None,
		embedder: DenseEmbedder | None = None,
		reranker: Any | None = None,
		metadata_processor: LegalMetadataProcessor | None = None,
		collection_name: str | None = None,
		dense_model_name: str | None = None,
		reranker_model_name: str | None = None,
		dense_candidate_limit: int | None = None,
		rerank_limit: int | None = None,
		expansion_article_limit: int = DEFAULT_EXPANSION_ARTICLE_LIMIT,
		max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
		relax_filter_on_empty: bool = True,
	):
		self.collection_name = collection_name or self._settings_value(
			"COLLECTION_NAME",
			self.DEFAULT_COLLECTION_NAME,
		)
		self.dense_model_name = dense_model_name or self._settings_value(
			"DENSE_EMBEDDING_MODEL",
			self.DEFAULT_DENSE_MODEL,
		)
		self.reranker_model_name = reranker_model_name or self._settings_value(
			"RERANKER_MODEL",
			self.DEFAULT_RERANKER_MODEL,
		)
		self.dense_candidate_limit = (
			dense_candidate_limit
			if dense_candidate_limit is not None
			else self._settings_value(
				"RETRIEVAL_CANDIDATE_LIMIT",
				self.DEFAULT_DENSE_CANDIDATE_LIMIT,
			)
		)
		self.rerank_limit = (
			rerank_limit
			if rerank_limit is not None
			else self._settings_value("RERANK_LIMIT", self.DEFAULT_RERANK_LIMIT)
		)
		self.expansion_article_limit = expansion_article_limit
		self.max_context_chars = max_context_chars
		self.relax_filter_on_empty = relax_filter_on_empty

		self.metadata_processor = metadata_processor or LegalMetadataProcessor()
		self.db_manager = db_manager or self._build_default_db_manager()
		self.embedder = embedder or self._build_default_embedder(self.dense_model_name)
		self.reranker = reranker

	def retrieve(self, query: str) -> dict[str, Any]:
		normalized_query = self.normalize_query(query)
		filters = self.extract_filters(normalized_query)

		if not normalized_query:
			return self._build_result(
				query=query,
				normalized_query=normalized_query,
				filters=filters,
				filter_applied=False,
				filter_relaxed=False,
				selected_chunks=[],
				expanded_chunks=[],
			)

		query_filter = self.build_query_filter(filters)
		filter_relaxed = False

		query_vector = self.embedder.embed_single(normalized_query)
		candidates = self.db_manager.query_dense(
			collection_name=self.collection_name,
			query_vector=query_vector,
			limit=self.dense_candidate_limit,
			query_filter=query_filter,
		)

		if not candidates and query_filter is not None and self.relax_filter_on_empty:
			filter_relaxed = True
			candidates = self.db_manager.query_dense(
				collection_name=self.collection_name,
				query_vector=query_vector,
				limit=self.dense_candidate_limit,
				query_filter=None,
			)

		dense_chunks = [
			self._format_chunk(candidate, source="dense")
			for candidate in candidates
		]
		selected_chunks = self.rerank(normalized_query, dense_chunks)
		expanded_chunks = self.expand_references(selected_chunks)

		return self._build_result(
			query=query,
			normalized_query=normalized_query,
			filters=filters,
			filter_applied=query_filter is not None and not filter_relaxed,
			filter_relaxed=filter_relaxed,
			selected_chunks=selected_chunks,
			expanded_chunks=expanded_chunks,
		)

	def normalize_query(self, query: str) -> str:
		if not isinstance(query, str):
			raise TypeError("query must be a string")
		return self.metadata_processor.normalize_text(query)

	def extract_filters(self, normalized_query: str) -> dict[str, Any]:
		extracted = self.metadata_processor.extract_references(normalized_query)
		return {
			"article_no": extracted.get("article_no"),
			"chapter_no": extracted.get("chapter_no"),
			"section_no": extracted.get("section_no"),
			"clause_nos": extracted.get("clause_nos", []),
			"ref_article_nos": extracted.get("ref_article_nos", []),
		}

	def build_query_filter(self, filters: dict[str, Any]) -> Filter | None:
		condition_specs = []

		for field_name in self.FILTER_FIELDS:
			value = filters.get(field_name)
			if value is not None:
				condition_specs.append((field_name, value))

		for clause_no in self._as_int_list(filters.get("clause_nos")):
			condition_specs.append(("clause_nos", clause_no))

		if not condition_specs:
			return None

		from qdrant_client.models import FieldCondition, Filter, MatchValue

		conditions = [
			FieldCondition(
				key=field_name,
				match=MatchValue(value=value),
			)
			for field_name, value in condition_specs
		]

		return Filter(must=conditions)

	def rerank(self, normalized_query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
		if not chunks or self.rerank_limit <= 0:
			return []

		pairs = [
			[normalized_query, self._chunk_text_for_matching(chunk)]
			for chunk in chunks
		]
		scores = self._predict_rerank_scores(pairs)

		reranked = []
		for chunk, score in zip(chunks, scores):
			reranked_chunk = dict(chunk)
			reranked_chunk["rerank_score"] = float(score)
			reranked_chunk["source"] = "reranked"
			reranked.append(reranked_chunk)

		reranked.sort(
			key=lambda chunk: (
				self._score_or_min(chunk.get("rerank_score")),
				self._score_or_min(chunk.get("dense_score")),
			),
			reverse=True,
		)
		return reranked[: self.rerank_limit]

	def expand_references(self, selected_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
		selected_ids = {
			chunk["id"]
			for chunk in selected_chunks
			if chunk.get("id") is not None
		}
		seen_ids = set(selected_ids)
		expanded_chunks = []

		for article_no in self._referenced_article_nos(selected_chunks):
			for fetched_chunk in self.db_manager.fetch_by_article_no(
				collection_name=self.collection_name,
				article_no=article_no,
				limit=self.expansion_article_limit,
			):
				chunk = self._format_chunk(
					fetched_chunk,
					source="reference",
					referenced_article_no=article_no,
				)
				chunk_id = chunk.get("id")
				if chunk_id is not None and chunk_id in seen_ids:
					continue
				if chunk_id is not None:
					seen_ids.add(chunk_id)
				expanded_chunks.append(chunk)

		return expanded_chunks

	def build_context(
		self,
		selected_chunks: list[dict[str, Any]],
		expanded_chunks: list[dict[str, Any]] | None = None,
	) -> str:
		expanded_chunks = expanded_chunks or []
		chunks = self._dedupe_chunks([*selected_chunks, *expanded_chunks])
		blocks = []
		total_chars = 0

		for index, chunk in enumerate(chunks, start=1):
			block = self._format_context_block(index, chunk)
			separator_len = 2 if blocks else 0
			next_total = total_chars + separator_len + len(block)

			if next_total <= self.max_context_chars:
				blocks.append(block)
				total_chars = next_total
				continue

			remaining = self.max_context_chars - total_chars - separator_len
			truncated_marker = "\n[đã rút gọn]"
			if remaining > len(truncated_marker):
				blocks.append(
					block[: remaining - len(truncated_marker)].rstrip()
					+ truncated_marker
				)
			break

		return "\n\n".join(blocks)

	def _build_result(
		self,
		query: str,
		normalized_query: str,
		filters: dict[str, Any],
		filter_applied: bool,
		filter_relaxed: bool,
		selected_chunks: list[dict[str, Any]],
		expanded_chunks: list[dict[str, Any]],
	) -> dict[str, Any]:
		context = self.build_context(selected_chunks, expanded_chunks)
		context_chunks = self._dedupe_chunks([*selected_chunks, *expanded_chunks])

		return {
			"query": query,
			"normalized_query": normalized_query,
			"filters": filters,
			"filter_applied": filter_applied,
			"filter_relaxed": filter_relaxed,
			"dense_candidate_limit": self.dense_candidate_limit,
			"rerank_limit": self.rerank_limit,
			"chunks": selected_chunks,
			"selected_chunks": selected_chunks,
			"expanded_chunks": expanded_chunks,
			"context_chunks": context_chunks,
			"context": context,
			"final_context": context,
		}

	def _predict_rerank_scores(self, pairs: list[list[str]]) -> list[float]:
		reranker = self._get_reranker()
		scores = reranker.predict(pairs)

		if hasattr(scores, "tolist"):
			scores = scores.tolist()
		if not isinstance(scores, list):
			scores = list(scores)
		return [float(score) for score in scores]

	def _get_reranker(self) -> Any:
		if self.reranker is None:
			from sentence_transformers import CrossEncoder

			self.reranker = CrossEncoder(self.reranker_model_name)
		return self.reranker

	def _settings_value(self, name: str, default: Any) -> Any:
		try:
			from src.configs import settings
		except Exception:
			return default
		return getattr(settings, name, default)

	def _build_default_db_manager(self) -> DBManager:
		from src.configs import settings
		from src.database import DBManager

		return DBManager(
			url=settings.QDRANT_URL,
			api_key=settings.QDRANT_API_KEY,
		)

	def _build_default_embedder(self, dense_model_name: str) -> DenseEmbedder:
		from src.data_pipeline import DenseEmbedder

		return DenseEmbedder(dense_model_name)

	def _format_chunk(
		self,
		chunk: dict[str, Any],
		source: str,
		referenced_article_no: int | None = None,
	) -> dict[str, Any]:
		metadata = dict(chunk.get("metadata") or chunk.get("payload") or {})
		content = chunk.get("content") or metadata.get("content", "")
		metadata.pop("content", None)

		formatted = {
			"id": str(chunk.get("id")) if chunk.get("id") is not None else None,
			"content": content,
			"metadata": metadata,
			"dense_score": chunk.get("dense_score", chunk.get("score")),
			"rerank_score": chunk.get("rerank_score"),
			"source": source,
		}
		if referenced_article_no is not None:
			formatted["referenced_article_no"] = referenced_article_no
		return formatted

	def _chunk_text_for_matching(self, chunk: dict[str, Any]) -> str:
		metadata = chunk.get("metadata") or {}
		metadata_text = " ".join(
			str(metadata.get(field_name, ""))
			for field_name in ("source", "chapter", "section", "article")
			if metadata.get(field_name)
		)
		normalized_source_text = metadata.get("normalized_source_text", "")
		content = chunk.get("content", "")
		return " ".join(
			part for part in (metadata_text, normalized_source_text, content) if part
		)

	def _format_context_block(self, index: int, chunk: dict[str, Any]) -> str:
		metadata = chunk.get("metadata") or {}
		lines = [f"[Đoạn {index}]"]

		for key, label in (
			("source", "Nguồn"),
			("chapter", "Chương"),
			("section", "Mục"),
			("article", "Điều"),
			("article_no", "Số điều"),
			("chapter_no", "Số chương"),
			("section_no", "Số mục"),
			("clause_nos", "Số khoản"),
			("ref_article_nos", "Số điều viện dẫn"),
		):
			value = metadata.get(key)
			if value not in (None, "", []):
				lines.append(f"{label}: {value}")

		if chunk.get("dense_score") is not None:
			lines.append(f"Điểm truy xuất dense: {chunk['dense_score']}")
		if chunk.get("rerank_score") is not None:
			lines.append(f"Điểm xếp hạng lại: {chunk['rerank_score']}")
		if chunk.get("source"):
			lines.append(f"Nguồn truy xuất: {chunk['source']}")
		if chunk.get("referenced_article_no") is not None:
			lines.append(f"Số điều được viện dẫn: {chunk['referenced_article_no']}")

		lines.append("Nội dung:")
		lines.append(str(chunk.get("content", "")))
		return "\n".join(lines)

	def _referenced_article_nos(self, chunks: list[dict[str, Any]]) -> list[int]:
		article_nos = []
		for chunk in chunks:
			metadata = chunk.get("metadata") or {}
			article_nos.extend(self._as_int_list(metadata.get("ref_article_nos")))
		return self._dedupe_ints(article_nos)

	def _dedupe_chunks(self, chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
		seen_ids = set()
		deduped = []
		for chunk in chunks:
			chunk_id = chunk.get("id")
			if chunk_id is not None:
				if chunk_id in seen_ids:
					continue
				seen_ids.add(chunk_id)
			deduped.append(chunk)
		return deduped

	def _as_int_list(self, value: Any) -> list[int]:
		if value in (None, ""):
			return []
		if isinstance(value, (list, tuple, set)):
			values = value
		else:
			values = [value]

		ints = []
		for item in values:
			try:
				ints.append(int(item))
			except (TypeError, ValueError):
				continue
		return self._dedupe_ints(ints)

	def _dedupe_ints(self, values: Iterable[int]) -> list[int]:
		seen = set()
		deduped = []
		for value in values:
			if value in seen:
				continue
			seen.add(value)
			deduped.append(value)
		return deduped

	def _score_or_min(self, value: Any) -> float:
		if value is None:
			return float("-inf")
		return float(value)
