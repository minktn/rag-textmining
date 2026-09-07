"""
Retrieval Pipeline
==================
Configurable retrieval pipeline for Vietnamese Legal RAG.

Supports three modes via ProcessingManager:
- Standard: Core dense retrieval → rerank → expand references
- Standard + Processing: Preprocessing → Core → Postprocessing
- Advanced: Replaces entire pipeline (e.g., RAG-Fusion)

Query rewriting (query_rewriter) is ALWAYS applied as the mandatory first step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from src.common import LegalMetadataProcessor
from src.config import settings

if TYPE_CHECKING:
	from qdrant_client.models import Filter

	from src.database.embedder import DenseEmbedder
	from src.database import DBManager
	from .processing_manager import ProcessingManager

logger = logging.getLogger(__name__)


class Retriever:
	"""Configurable retrieval pipeline for legal RAG queries.

	Pipeline flow:
	1. query_rewriter (ALWAYS first — mandatory)
	2. normalize_query + extract_filters
	3. Dispatch → advanced OR (preprocessing → core retrieval → postprocessing)
	"""

	DEFAULT_COLLECTION_NAME = "landlaw"
	DEFAULT_DENSE_MODEL = "BAAI/bge-m3"
	DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
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
		processing_manager: ProcessingManager | None = None,
		sub_llm_manager: Any | None = None,
	):
		self.collection_name = collection_name or self._settings_value(
			"COLLECTION_NAME",
			self.DEFAULT_COLLECTION_NAME,
		)
		self.dense_model_name = dense_model_name or self._settings_value(
			"EMBEDDING_MODEL",
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
		self.sub_llm_manager = sub_llm_manager

		# Processing pipeline configuration
		if processing_manager is not None:
			self.processing = processing_manager
		else:
			from .processing_manager import ProcessingManager
			self.processing = ProcessingManager.from_settings()

		logger.info(f"[Retriever] Initialized with {self.processing}")

	# ══════════════════════════════════════════════════════════════
	# Main Entry Point
	# ══════════════════════════════════════════════════════════════

	def retrieve(self, query: str) -> dict[str, Any]:
		"""Main retrieval entry point.

		Flow:
		1. query_rewriter (always first)
		2. normalize + extract filters
		3. Dispatch: advanced → _retrieve_advanced
		                  else → _retrieve_standard (with optional pre/post processing)
		"""
		# Step 1: Query rewriting (ALWAYS mandatory first step)
		rewritten_query = self._rewrite_query(query)

		# Step 2: Normalize and extract metadata filters
		normalized_query = self.normalize_query(rewritten_query)
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

		# Step 3: Dispatch based on processing configuration
		if self.processing.advanced:
			return self._retrieve_advanced(query, normalized_query, filters)

		return self._retrieve_standard(query, normalized_query, filters)

	# ══════════════════════════════════════════════════════════════
	# Query Rewriting (mandatory first step)
	# ══════════════════════════════════════════════════════════════

	def _rewrite_query(self, query: str) -> str:
		"""Always run query_rewriter as the first step. Graceful fallback on failure."""
		try:
			from src.retriever.processing.preprocessing.query_rewriter import rewrite_query
			rewritten = rewrite_query(query, llm=self.sub_llm_manager)
			if rewritten and rewritten != query:
				logger.info(f"[Retriever] Query rewritten: '{query[:60]}' → '{rewritten[:60]}'")
			return rewritten
		except ImportError:
			logger.debug("[Retriever] query_rewriter not available, using original query.")
			return query
		except Exception as e:
			logger.warning(f"[Retriever] Query rewrite failed: {e}. Using original query.")
			return query

	# ══════════════════════════════════════════════════════════════
	# Advanced Retrieval (replaces entire standard pipeline)
	# ══════════════════════════════════════════════════════════════

	# ══════════════════════════════════════════════════════════════
	# Advanced Retrieval (Strategy Pattern - Zero hardcoded methods)
	# ══════════════════════════════════════════════════════════════

	def _retrieve_advanced(
		self, query: str, normalized_query: str, filters: dict[str, Any],
	) -> dict[str, Any]:
		"""Dispatch to advanced retrieval strategy via dynamic factory."""
		method = self.processing.advanced
		logger.info(f"[Retriever] Advanced retrieval strategy: '{method}'")

		from src.retriever.advanced import get_advanced_retriever

		strategy = get_advanced_retriever(
			method,
			sub_llm_manager=self.sub_llm_manager,
		)
		return strategy.retrieve(
			query=query,
			retriever=self,
			normalized_query=normalized_query,
			filters=filters,
		)

	# ══════════════════════════════════════════════════════════════
	# Standard Retrieval (with optional pre/post processing)
	# ══════════════════════════════════════════════════════════════

	def _retrieve_standard(
		self, query: str, normalized_query: str, filters: dict[str, Any],
	) -> dict[str, Any]:
		"""Standard pipeline: preprocessing → dense search + rerank → postprocessing."""
		query_filter = self.build_query_filter(filters)
		filter_relaxed = False

		# ── Preprocessing (Uỷ quyền hoàn toàn cho ProcessingManager) ───
		processed_query = self.processing.apply_preprocessing(
			query=normalized_query,
			retriever=self,
		)

		self._current_query = processed_query

		# ── Core dense retrieval (Priority: Local > Cloud) ─────
		query_vector = getattr(self, "_hyde_embedding", None) or self.embedder.embed_single(processed_query)
		self._hyde_embedding = None

		candidates = self._query_dense_candidates(
			query_vector=query_vector,
			query_filter=query_filter,
		)

		if not candidates and query_filter is not None and self.relax_filter_on_empty:
			filter_relaxed = True
			candidates = self._query_dense_candidates(
				query_vector=query_vector,
				query_filter=None,
			)

		dense_chunks = [
			self._format_chunk(candidate, source="dense")
			for candidate in candidates
		]
		selected_chunks = self.rerank(processed_query, dense_chunks)

		# ── Postprocessing (Uỷ quyền hoàn toàn cho ProcessingManager) ──
		selected_chunks = self.processing.apply_postprocessing(
			query=query,
			chunks=selected_chunks,
			retriever=self,
		)

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

	def _query_dense_candidates(
		self,
		query_vector: list[float],
		query_filter: Any = None,
	) -> list[dict[str, Any]]:
		"""Truy vấn ứng viên dense: Ưu tiên Local vector index > Cloud Qdrant."""
		if getattr(settings, "PREFER_LOCAL_STORAGE", True):
			try:
				is_graph = (
					self.collection_name in ("graph", "neo4j", "graphrag")
					or self.dense_model_name in ("graph", "neo4j", "graphrag")
				)
				is_contriever = (
					self.collection_name == getattr(settings, "CONTRIEVER_COLLECTION_NAME", "landlaw_contriever")
					or self.dense_model_name == getattr(settings, "CONTRIEVER_MODEL", "facebook/mcontriever-msmarco")
				)
				if is_graph:
					from src.database.storage.graph_store import Neo4jGraphStore
					graph_store = Neo4jGraphStore()
					return graph_store.query_chunks(
						query=getattr(self, "_current_query", ""),
						limit=self.dense_candidate_limit,
						query_vector=query_vector,
					)
				elif is_contriever:
					from src.database.storage.contriever_store import ContrieverStore
					store = ContrieverStore(
						collection_name=self.collection_name,
						db_manager=self.db_manager,
					)
				else:
					from src.database.storage.base_store import BaseStore
					store = BaseStore(
						collection_name=self.collection_name,
						db_manager=self.db_manager,
					)

				if store.is_local_available():
					dict_filter = query_filter if isinstance(query_filter, dict) else None
					return store.query_local(
						query_vector,
						limit=self.dense_candidate_limit,
						query_filter=dict_filter,
					)
			except Exception as e:
				logger.warning(f"Lỗi truy vấn local index ({e}). Chuyển sang Qdrant Cloud.")

		return self.db_manager.query_dense(
			collection_name=self.collection_name,
			query_vector=query_vector,
			limit=self.dense_candidate_limit,
			query_filter=query_filter,
		)

	# ══════════════════════════════════════════════════════════════
	# Core methods (unchanged from original)
	# ══════════════════════════════════════════════════════════════

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
			try:
				fetched_chunks = self._fetch_by_article_no(article_no)
			except Exception as e:
				logger.warning(f"[Retriever] Không thể fetch references cho Điều {article_no}: {e}")
				fetched_chunks = []

			for fetched_chunk in fetched_chunks:
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

	def _fetch_by_article_no(self, article_no: int) -> list[dict[str, Any]]:
		"""Tìm kiếm các chunks theo Điều luật article_no: Ưu tiên Local chunks > Qdrant Cloud."""
		if getattr(settings, "PREFER_LOCAL_STORAGE", True):
			try:
				chunks = self._fetch_local_by_article_no(article_no)
				if chunks:
					return chunks
			except Exception as e:
				logger.warning(f"[Retriever] Local fetch by article {article_no} error: {e}. Fallback to Cloud.")

		return self.db_manager.fetch_by_article_no(
			collection_name=self.collection_name,
			article_no=article_no,
			limit=self.expansion_article_limit,
		)

	def _fetch_local_by_article_no(self, article_no: int) -> list[dict[str, Any]]:
		"""Trích xuất chunks từ file local landlaw_chunks.json theo article_no."""
		if not hasattr(self, "_local_article_index"):
			index: dict[int, list[dict[str, Any]]] = {}
			chunks_path = Path(settings.CHUNKED_DATA_DIR / "landlaw_chunks.json")
			if chunks_path.exists():
				import json
				with open(chunks_path, "r", encoding="utf-8") as f:
					chunks = json.load(f)
				for c in chunks:
					meta = c.get("metadata") or {}
					art = meta.get("article_no")
					if art is not None:
						index.setdefault(art, []).append(c)
			self._local_article_index = index

		matching = self._local_article_index.get(article_no, [])
		return matching[: self.expansion_article_limit]

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

	# ══════════════════════════════════════════════════════════════
	# Private helpers (unchanged)
	# ══════════════════════════════════════════════════════════════

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
		return getattr(settings, name, default)

	def _build_default_db_manager(self) -> DBManager:
		from src.database import DBManager

		return DBManager(
			url=settings.QDRANT_URL,
			api_key=settings.QDRANT_API_KEY,
		)

	def _build_default_embedder(self, dense_model_name: str) -> DenseEmbedder:
		from src.database.embedder import DenseEmbedder

		return DenseEmbedder(dense_model_name)

	def _format_chunk(
		self,
		chunk: dict[str, Any],
		source: str,
		referenced_article_no: int | None = None,
	) -> dict[str, Any]:
		raw_metadata = dict(chunk.get("metadata") or chunk.get("payload") or {})
		content = chunk.get("content") or raw_metadata.get("content", "")
		raw_metadata.pop("content", None)

		# Luôn làm giàu (enrich) metadata qua LegalMetadataProcessor
		metadata = self.metadata_processor.enrich_payload(raw_metadata, content)

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
