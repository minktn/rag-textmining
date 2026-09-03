"""
RAG Evaluator — Pipeline Execution & Orchestration
===================================================
Điều phối toàn bộ quy trình: Load Dataset -> Retrieve -> Generate -> Compute Metrics -> Report.

Đồng bộ hoá toàn diện cho UI & API:
  - Tiếp nhận mọi cấu hình từ UI: retriever_mode ('base', 'contriever', 'graph'),
    llm_service, sub_llm_service, advanced_method ('rag_fusion'), preprocessing, postprocessing.
  - Tự động đọc và đóng gói 'metadata' phản ánh chính xác cấu hình thực thi từ các module.
  - Chuẩn hóa JSON output phục vụ hiển thị động trên UI (Badges, Alert Banner, Metrics Cards).
"""

from datetime import datetime
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import settings
from src.generation import LLMManager, SubLLMManager
from src.retriever import ProcessingManager, Retriever
from .data_loader import EvalDataLoader
from .metrics import MetricsCalculator
from .ragas_metrics import RagasJudge
from .reporting import EvaluationReporter
from .text_processing import safe_print

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """OOP Orchestrator điều phối toàn bộ pipeline đánh giá RAG và đồng bộ hoá metadata cho UI."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_manager: Optional[LLMManager] = None,
        sub_llm_manager: Optional[SubLLMManager] = None,
        ragas_judge: Optional[RagasJudge] = None,
        reporter: Optional[EvaluationReporter] = None,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        # Các lựa chọn từ UI
        retriever_mode: str = "base",  # "base" | "contriever" | "graph"
        llm_service: Optional[str] = None,  # "nvidia" | "groq" | "google" | "local"
        llm_mode: Optional[str] = None,  # "reason" | "base"
        sub_llm_service: Optional[str] = None,  # "local" | "nvidia" | "groq" | "google"
        sub_llm_mode: Optional[str] = None,
        processing_manager: Optional[ProcessingManager] = None,
        advanced: Optional[str] = None,  # "rag_fusion" | ""
        preprocessing: Optional[List[str]] = None,
        postprocessing: Optional[List[str]] = None,
        ragas_service: Optional[str] = None,  # "nvidia" | "groq" | "google"
        # Graph & Storage params
        use_graph: Optional[bool] = None,
        graph_method: str = "local",
        root_dir: Optional[Path] = None,
        embedder: Optional[Any] = None,
        db_manager: Optional[Any] = None,
    ):
        # 1. Xác định chế độ Retriever
        if use_graph is not None:
            self.use_graph = use_graph
            self.retriever_mode = "graph" if use_graph else "base"
        else:
            self.retriever_mode = (retriever_mode or "base").lower().strip()
            self.use_graph = (self.retriever_mode == "graph")

        self.graph_method = graph_method
        self.top_k = top_k if top_k is not None else getattr(settings, "RERANK_LIMIT", 5)
        self.root_dir = Path(root_dir or settings.GRAPH_DB_DIR)

        # 2. Xác định collection theo retriever_mode
        if collection_name:
            self.collection_name = collection_name
        elif self.retriever_mode == "contriever":
            self.collection_name = getattr(settings, "CONTRIEVER_COLLECTION_NAME", "landlaw_contriever")
        else:
            self.collection_name = getattr(settings, "COLLECTION_NAME", "landlaw")

        # 3. Khởi tạo LLMs
        self.llm_manager = llm_manager or LLMManager(service=llm_service, mode=llm_mode)
        self.sub_llm_manager = sub_llm_manager or SubLLMManager(service=sub_llm_service, mode=sub_llm_mode)
        self.model_name = model_name or self.llm_manager.get_default_model(self.llm_manager.service)

        self.ragas_judge = ragas_judge or RagasJudge(service=ragas_service)
        self.reporter = reporter or EvaluationReporter()

        from src.common.legal_metadata import LegalMetadataProcessor
        self.metadata_processor = LegalMetadataProcessor()

        # 4. Khởi tạo ProcessingManager (chuẩn OOP, không dùng flag rời rạc)
        if processing_manager is not None:
            self.processing_manager = processing_manager
        elif advanced is not None or preprocessing is not None or postprocessing is not None:
            self.processing_manager = ProcessingManager(
                preprocessing=preprocessing,
                postprocessing=postprocessing,
                advanced=advanced,
            )
        else:
            self.processing_manager = ProcessingManager.from_settings()

        # 5. Khởi tạo Retriever
        if retriever is not None:
            self.retriever = retriever
            self.retriever.rerank_limit = self.top_k
        else:
            if self.retriever_mode == "graph":
                dense_model = settings.EMBEDDING_MODEL  # BAAI/bge-m3 (1024 chiều cho Graph DB)
                coll_name = "graph"
            elif self.retriever_mode == "contriever":
                dense_model = settings.CONTRIEVER_MODEL  # Contriever (768 chiều)
                coll_name = getattr(settings, "CONTRIEVER_COLLECTION_NAME", "landlaw_contriever")
            else:
                dense_model = settings.EMBEDDING_MODEL  # BAAI/bge-m3 (1024 chiều cho Base)
                coll_name = self.collection_name
            self.retriever = Retriever(
                db_manager=db_manager,
                embedder=embedder,
                collection_name=coll_name,
                dense_model_name=dense_model,
                rerank_limit=self.top_k,
                processing_manager=self.processing_manager,
                sub_llm_manager=self.sub_llm_manager,
            )

    # ─────────────────────────────────────────────────────────────
    # Metadata Introspection for UI Synchronization
    # ─────────────────────────────────────────────────────────────

    def get_pipeline_metadata(self) -> Dict[str, Any]:
        """Tự động đọc thông tin thực tế từ tất cả các module trong pipeline phục vụ hiển thị UI."""
        return {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "retriever_mode": self.retriever_mode,
                "advanced_method": self.processing_manager.advanced or None,
                "preprocessing": self.processing_manager.preprocessing,
                "postprocessing": self.processing_manager.postprocessing,
                "llm_service": self.llm_manager.service,
                "llm_model": self.model_name,
                "llm_mode": self.llm_manager.mode,
                "sub_llm_service": self.sub_llm_manager.service if self.sub_llm_manager else None,
                "sub_llm_model": (
                    self.sub_llm_manager.get_default_model(self.sub_llm_manager.service)
                    if self.sub_llm_manager
                    else None
                ),
                "sub_llm_mode": self.sub_llm_manager.mode if self.sub_llm_manager else None,
                "embedding_model": settings.EMBEDDING_MODEL,
                "collection_name": self.collection_name,
                "top_k": self.top_k,
                "ragas_service": self.ragas_judge.service if self.ragas_judge else None,
                "ragas_model": self.ragas_judge.model_name if self.ragas_judge else None,
            },
            "is_graph_mode": self.use_graph,
            "notice": (
                "Graph Database Mode: Các chỉ số truy xuất theo chunk thô (Recall@K, Precision@K, MRR, nDCG) là N/A. Đánh giá dựa trên RAGAS (LLM-as-a-judge) & Generation."
                if self.use_graph
                else None
            ),
        }

    # ─────────────────────────────────────────────────────────────
    # Retrieval Phase
    # ─────────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> Dict[str, Any]:
        """Truy vấn contexts + payloads sử dụng Retriever chính thức của hệ thống."""
        t0 = time.perf_counter()

        if self.retriever is not None:
            res = self.retriever.retrieve(question)

            chunks = res.get("chunks") or res.get("selected_chunks") or []
            if self.top_k and len(chunks) > self.top_k:
                chunks = chunks[: self.top_k]

            contexts = [c.get("content", "") for c in chunks]
            payloads = []
            for c in chunks:
                meta = dict(c.get("metadata") or {})
                if "article_no" not in meta or meta.get("article_no") is None:
                    meta = self.metadata_processor.enrich_payload(meta, c.get("content", ""))
                payloads.append({
                    **meta,
                    "content": c.get("content", ""),
                    "id": c.get("id"),
                    "dense_score": c.get("dense_score"),
                    "rerank_score": c.get("rerank_score"),
                    "source": c.get("source"),
                })

            docs = chunks
            full_context = res.get("context", "")
        else:
            chunks, contexts, payloads, docs, full_context = [], [], [], [], ""

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "contexts": contexts,
            "payloads": payloads,
            "docs": docs,
            "full_context": full_context,
            "latency_ms": round(latency_ms, 2),
        }

    # ─────────────────────────────────────────────────────────────
    # Generation Phase
    # ─────────────────────────────────────────────────────────────

    def generate(self, question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sinh câu trả lời từ LLM với danh sách chunks truyền vào."""
        t0 = time.perf_counter()
        prompt = self.llm_manager.construct_prompt(question, docs=docs)
        answer = self.llm_manager.generate_response(prompt, model_name=self.model_name) or ""
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "answer": answer,
            "latency_ms": round(latency_ms, 2),
        }

    # ─────────────────────────────────────────────────────────────
    # GraphRAG Execution
    # ─────────────────────────────────────────────────────────────

    def run_graphrag(self, question: str) -> Dict[str, Any]:
        """Thực thi truy vấn GraphRAG engine để lấy câu trả lời và context tri thức."""
        t0 = time.perf_counter()
        cmd = [
            "uv", "run", "graphrag", "query",
            "--root", str(self.root_dir),
            "--method", self.graph_method,
            question,
        ]

        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=180,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            if res.returncode != 0:
                error_msg = res.stderr or res.stdout
                logger.warning(f"[GraphRAG Warning] Lỗi khi chạy CLI: {error_msg.strip()}")
                answer = f"[GraphRAG Error] {error_msg.strip()}"
                contexts = [answer]
            else:
                answer = res.stdout.strip()
                contexts = [answer]
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"[GraphRAG Error] Ngoại lệ: {e}")
            answer = f"[GraphRAG Exception] {e}"
            contexts = [answer]

        payloads = [MetricsCalculator.extract_payload_from_text(ctx) for ctx in contexts]
        return {
            "answer": answer,
            "contexts": contexts,
            "payloads": payloads,
            "latency_ms": round(latency_ms, 2),
        }

    # ─────────────────────────────────────────────────────────────
    # Single Question Evaluation
    # ─────────────────────────────────────────────────────────────

    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Thực hiện retrieve + generate cho 1 câu hỏi."""
        question = item["question"]

        retrieval = self.retrieve(question)
        generation = self.generate(question, retrieval["docs"])
        answer = generation["answer"]
        contexts = retrieval["contexts"]
        payloads = retrieval["payloads"]
        retrieval_latency = retrieval["latency_ms"]
        generation_latency = generation["latency_ms"]

        retrieved_law_ids = MetricsCalculator.extract_law_ids_from_payloads(payloads)

        return {
            "id": item.get("id"),
            "question": question,
            "question_type": item.get("question_type", ""),
            "ground_truth": item.get("answer", ""),
            "generated_answer": answer,
            "retrieved_contexts": contexts,
            "retrieved_payloads": payloads,
            "retrieved_law_ids": retrieved_law_ids,
            "law_id": item.get("law_id", {}),
            "is_graph": self.use_graph,
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "generation_latency_ms": round(generation_latency, 2),
            "e2e_latency_ms": round(retrieval_latency + generation_latency, 2),
        }

    # ─────────────────────────────────────────────────────────────
    # Batch / All Evaluation
    # ─────────────────────────────────────────────────────────────

    def evaluate_all(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Đánh giá toàn bộ danh sách câu hỏi với tiến trình hiển thị an toàn."""
        results = []
        total = len(questions)

        for idx, item in enumerate(questions, 1):
            qid = item.get("id", f"Q{idx}")
            q_preview = item.get("question", "")[:60]
            safe_print(f"\r  [{idx}/{total}] {qid}: {q_preview}...", end="", flush=True)

            result = self.evaluate_single(item)
            results.append(result)

        safe_print()
        return results

    # ─────────────────────────────────────────────────────────────
    # Full Workflow Orchestration & Standardized JSON Output
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        eval_file: Optional[Path] = None,
        limit: Optional[int] = None,
        random_sample: bool = False,
        seed: int = 42,
        skip_ragas: bool = False,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ quy trình đánh giá hoàn chỉnh và trả về cấu trúc JSON chuẩn hóa
        đồng bộ 100% với lựa chọn trên UI.
        """
        # 1. Load data
        loader = EvalDataLoader()
        dataset_meta, questions = loader.load(
            eval_file=eval_file,
            limit=limit,
            random_sample=random_sample,
            seed=seed,
        )

        if not questions:
            safe_print("Không có câu hỏi nào để đánh giá!")
            return {}

        safe_print(f"\n[Phase 1] Đang chạy RAG pipeline ({self.retriever_mode}) cho {len(questions)} câu hỏi...")
        results = self.evaluate_all(questions)

        safe_print("\n[Phase 2] Tính toán các chỉ số cơ bản (Basic Metrics)...")
        basic_metrics = MetricsCalculator.aggregate(
            results,
            top_k=self.top_k,
            is_graph_mode=self.use_graph,
        )

        ragas_scores = {}
        if not skip_ragas:
            safe_print("\n[Phase 3] Tính toán RAGAS Metrics (LLM-as-a-judge)...")
            ragas_scores = self.ragas_judge.evaluate(results)
        else:
            safe_print("\n[Phase 3] Bỏ qua RAGAS Metrics (--skip-ragas).")

        # 2. In bảng kết quả
        self.reporter.print_summary_table(basic_metrics, ragas_scores, self.model_name)

        # 3. Đóng gói JSON chuẩn hóa đầy đủ metadata cho UI
        pipeline_meta = self.get_pipeline_metadata()
        pipeline_meta["eval_dataset"] = {
            "law_name": dataset_meta.get("law_name", "Luật Đất đai 2024"),
            "law_number": dataset_meta.get("law_number", "31/2024/QH15"),
            "total_questions": len(questions),
            "version": dataset_meta.get("version", "2.1"),
        }

        # Clean payloads để lưu JSON an toàn
        clean_results = []
        for r in results:
            clean_item = {k: v for k, v in r.items() if k != "retrieved_payloads"}
            clean_results.append(clean_item)

        unified_output = {
            "metadata": pipeline_meta,
            "summary_metrics": {
                "is_graph_mode": self.use_graph,
                "retrieval": {
                    "hit_rate": basic_metrics.get("retrieval_hit_rate"),
                    "mrr": basic_metrics.get("mrr"),
                    "recall_at_k": basic_metrics.get("avg_recall_at_k"),
                    "precision_at_k": basic_metrics.get("avg_precision_at_k"),
                    "ndcg": basic_metrics.get("avg_ndcg"),
                    "latency_ms": basic_metrics.get("avg_retrieval_latency_ms"),
                    "status": "N/A (Graph Mode - Entity/Community context)" if self.use_graph else "OK",
                },
                "generation": {
                    "exact_match_rate": basic_metrics.get("exact_match_rate"),
                    "avg_f1": basic_metrics.get("avg_f1"),
                    "avg_bleu_1": basic_metrics.get("avg_bleu_1"),
                    "avg_rouge_l": basic_metrics.get("avg_rouge_l"),
                    "latency_ms": basic_metrics.get("avg_generation_latency_ms"),
                    "e2e_latency_ms": basic_metrics.get("avg_e2e_latency_ms"),
                },
                "ragas": ragas_scores,
                "per_question_type": basic_metrics.get("per_question_type", {}),
            },
            "detailed_results": clean_results,
        }

        # 4. Lưu file JSON
        if save:
            self.reporter.save_unified_report(unified_output)

        return unified_output
