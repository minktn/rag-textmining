"""
RAG Evaluator — Pipeline Execution
====================================
Orchestrate retrieve → generate → evaluate cho từng câu hỏi.
Hỗ trợ các chế độ:
  - Standard RAG (Qdrant + LLM)
  - RAG-Fusion (Multi-Query Expansion + RRF)
  - GraphRAG (Microsoft Knowledge Graph RAG)
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.database import DBManager
from src.database.embedder import DenseEmbedder
from src.evaluation.text_processing import safe_print
from src.generation import LLMManager

print = safe_print


class RAGEvaluator:
	"""Orchestrator cho toàn bộ RAG evaluation pipeline."""

	def __init__(
		self,
		embedder: DenseEmbedder,
		db_manager: DBManager,
		llm_manager: LLMManager,
		model_name: str,
		collection_name: str = "landlaw",
		top_k: int = 5,
		use_graph: bool = False,
		use_rag_fusion: bool = False,
		graph_method: str = "local",
		root_dir: Optional[Path] = None,
	):
		self.embedder = embedder
		self.db_manager = db_manager
		self.llm_manager = llm_manager
		self.model_name = model_name
		self.collection_name = collection_name
		self.top_k = top_k
		self.use_graph = use_graph
		self.use_rag_fusion = use_rag_fusion
		self.graph_method = graph_method
		self.root_dir = root_dir or Path(__file__).resolve().parents[3]

		if self.use_rag_fusion:
			from src.retriever.processing.preprocessing.rag_fusion import RAGFusionProcessor
			self.fusion_processor = RAGFusionProcessor(k=60)
		else:
			self.fusion_processor = None

	def _query_points_single(self, question: str) -> List[Dict[str, Any]]:
		"""Helper gọi Qdrant trả về danh sách dict tài liệu."""
		query_vector = self.embedder.embed_single(question)
		raw_results = self.db_manager.client.query_points(
			collection_name=self.collection_name,
			query=query_vector,
			using="dense",
			limit=self.top_k,
		)
		docs = []
		for point in raw_results.points:
			payload = dict(point.payload or {})
			payload["id"] = str(point.id)
			docs.append(payload)
		return docs

	# ── Retrieve ──────────────────────────────────────────────────

	def retrieve(self, question: str) -> dict:
		"""Truy vấn contexts + payloads + latency (Standard hoặc RAG-Fusion)."""
		t0 = time.perf_counter()

		if self.use_rag_fusion and self.fusion_processor:
			all_queries, fused_docs = self.fusion_processor.process(
				query=question,
				retrieve_fn=self._query_points_single,
				num_queries=4,
				diverse=True,
				original_query_weight=3.0,
				top_k=self.top_k,
			)
			contexts = [doc.get("content", "") for doc in fused_docs]
			payloads = fused_docs
			docs = [{"content": c} for c in contexts]
		else:
			raw_docs = self._query_points_single(question)
			contexts = [doc.get("content", "") for doc in raw_docs]
			payloads = raw_docs
			docs = [{"content": c} for c in contexts]

		latency_ms = (time.perf_counter() - t0) * 1000

		return {
			"contexts": contexts,
			"payloads": payloads,
			"docs": docs,
			"latency_ms": round(latency_ms, 2),
		}

	# ── Generate ──────────────────────────────────────────────────

	def generate(self, question: str, docs: list[dict]) -> dict:
		"""Generate answer từ LLM, trả về answer + latency."""
		t0 = time.perf_counter()
		prompt = self.llm_manager.construct_prompt(question, docs=docs)
		answer = self.llm_manager.generate_response(prompt, model_name=self.model_name)
		latency_ms = (time.perf_counter() - t0) * 1000

		if answer is None:
			answer = ""

		return {
			"answer": answer,
			"latency_ms": round(latency_ms, 2),
		}

	# ── GraphRAG Execution ───────────────────────────────────────

	def run_graphrag(self, question: str) -> dict:
		"""Chạy GraphRAG query engine qua CLI để thu về answer + contexts."""
		from .metrics import MetricsCalculator

		t0 = time.perf_counter()
		cmd = [
			"uv", "run", "graphrag", "query",
			"--root", str(self.root_dir),
			"--method", self.graph_method,
			question
		]

		try:
			res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
			latency_ms = (time.perf_counter() - t0) * 1000

			if res.returncode != 0:
				error_msg = res.stderr or res.stdout
				print(f"\n  [GraphRAG Warning] Lỗi khi chạy CLI: {error_msg.strip()}")
				answer = f"[GraphRAG Error] {error_msg.strip()}"
				contexts = [answer]
			else:
				answer = res.stdout.strip()
				contexts = [answer]

		except Exception as e:
			latency_ms = (time.perf_counter() - t0) * 1000
			print(f"\n  [GraphRAG Error] Exception: {e}")
			answer = f"[GraphRAG Exception] {e}"
			contexts = [answer]

		payloads = [MetricsCalculator.extract_payload_from_text(ctx) for ctx in contexts]

		return {
			"answer": answer,
			"contexts": contexts,
			"payloads": payloads,
			"latency_ms": round(latency_ms, 2),
		}

	# ── Single question ───────────────────────────────────────────

	def evaluate_single(self, item: dict) -> dict:
		"""Chạy retrieve + generate cho 1 câu hỏi, trả về kết quả chi tiết."""
		from .metrics import MetricsCalculator

		question = item["question"]

		if self.use_graph:
			graph_res = self.run_graphrag(question)
			answer = graph_res["answer"]
			contexts = graph_res["contexts"]
			payloads = graph_res["payloads"]
			retrieval_latency = graph_res["latency_ms"] / 2
			generation_latency = graph_res["latency_ms"] / 2
		else:
			retrieval = self.retrieve(question)
			generation = self.generate(question, retrieval["docs"])
			answer = generation["answer"]
			contexts = retrieval["contexts"]
			payloads = retrieval["payloads"]
			retrieval_latency = retrieval["latency_ms"]
			generation_latency = generation["latency_ms"]

		retrieved_law_ids = MetricsCalculator.extract_law_ids_from_payloads(payloads)

		return {
			"id": item["id"],
			"question": question,
			"question_type": item.get("question_type", ""),
			"ground_truth": item["answer"],
			"generated_answer": answer,
			"retrieved_contexts": contexts,
			"retrieved_payloads": payloads,
			"retrieved_law_ids": retrieved_law_ids,
			"law_id": item.get("law_id", {}),
			"retrieval_latency_ms": round(retrieval_latency, 2),
			"generation_latency_ms": round(generation_latency, 2),
			"e2e_latency_ms": round(retrieval_latency + generation_latency, 2),
		}

	# ── Full dataset ──────────────────────────────────────────────

	def evaluate_all(self, questions: list[dict]) -> list[dict]:
		"""Chạy pipeline cho toàn bộ dataset với progress indicator."""
		results = []
		total = len(questions)

		for idx, item in enumerate(questions, 1):
			qid = item["id"]
			print(f"\r  [{idx}/{total}] {qid}: {item['question'][:60]}...", end="", flush=True)

			result = self.evaluate_single(item)
			results.append(result)

		print()  # newline after progress
		return results
