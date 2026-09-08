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
        # Batch & Concurrency params
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        ragas_max_workers: Optional[int] = None,
        ragas_batch_size: Optional[int] = None,
        skip_base: bool = False,
    ):
        self.skip_base = skip_base

        # Batch & Concurrency
        self.batch_size = batch_size or getattr(settings, "EVAL_BATCH_SIZE", 10)
        self.max_workers = max_workers or getattr(settings, "EVAL_MAX_WORKERS", 4)
        self.ragas_max_workers = ragas_max_workers or getattr(settings, "RAGAS_MAX_WORKERS", 4)
        self.ragas_batch_size = ragas_batch_size if ragas_batch_size is not None else getattr(settings, "RAGAS_BATCH_SIZE", self.ragas_max_workers)

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

        self.ragas_judge = ragas_judge or RagasJudge(
            service=ragas_service,
            batch_size=self.ragas_batch_size,
            max_workers=self.ragas_max_workers,
        )
        self.reporter = reporter or EvaluationReporter()

        from src.common.legal_metadata import LegalMetadataProcessor
        self.metadata_processor = LegalMetadataProcessor()

        # 4 & 5. Khởi tạo ProcessingManager & Retriever (BỎ QUA KHI SKIP_BASE)
        if self.skip_base:
            # Khi --skip-base được kích hoạt: Tuyệt đối KHÔNG load model Retriever/Embedder/Filter lên GPU
            class DummyProcessing:
                def __init__(self, adv, pre, post):
                    self.advanced = adv
                    self.preprocessing = pre or []
                    self.postprocessing = post or []
                def __str__(self):
                    return f"DummyProcessing(advanced={self.advanced}, pre={self.preprocessing}, post={self.postprocessing})"
            self.processing_manager = DummyProcessing(advanced, preprocessing, postprocessing)
            self.retriever = None
            safe_print("  [Init] Chế độ --skip-base: KHÔNG load Retriever, Embedder hay model GPU nào của session này.")
        else:
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

    def get_pipeline_metadata(
        self,
        eval_file: Optional[Union[str, Path]] = None,
        limit: Optional[int] = None,
        random_sample: bool = False,
        seed: int = 42,
        skip_ragas: bool = False,
        skip_base: bool = False,
    ) -> Dict[str, Any]:
        """Tự động đọc thông tin thực tế từ tất cả các module trong pipeline phục vụ hiển thị UI & checkpointing."""
        eval_file_name = Path(eval_file).name if eval_file else "eval_landlaw_2024.json"
        return {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "eval_file": eval_file_name,
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
                "reranker_model": getattr(settings, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                "candidate_limit": getattr(settings, "RETRIEVAL_CANDIDATE_LIMIT", 20),
                "device": getattr(settings, "DEVICE", "cpu"),
                "collection_name": self.collection_name,
                "top_k": self.top_k,
                "limit": limit,
                "random_sample": random_sample,
                "seed": seed,
                "skip_ragas": skip_ragas,
                "skip_base": skip_base,
                "graph_method": self.graph_method if self.use_graph else None,
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
                "ragas_max_workers": self.ragas_max_workers,
                "ragas_batch_size": self.ragas_batch_size,
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
        """Thực hiện retrieve + generate cho 1 câu hỏi và tính ngay toàn bộ metrics cho case này."""
        question = item["question"]
        ground_truth = item.get("answer", "") or item.get("ground_truth", "")

        retrieval = self.retrieve(question)
        generation = self.generate(question, retrieval["docs"])
        answer = generation["answer"]
        contexts = retrieval["contexts"]
        payloads = retrieval["payloads"]
        retrieval_latency = retrieval["latency_ms"]
        generation_latency = generation["latency_ms"]

        retrieved_law_ids = MetricsCalculator.extract_law_ids_from_payloads(payloads)

        # Tính ngay toàn bộ Retrieval Metrics cho từng case
        if not self.use_graph:
            ret_metrics = MetricsCalculator.compute_retrieval_metrics(payloads, item.get("law_id", {}))
        else:
            ret_metrics = {
                "retrieval_hit": None,
                "mrr": None,
                "recall_at_k": None,
                "precision_at_k": None,
                "ndcg": None,
            }

        # Tính ngay toàn bộ Generation Metrics cho từng case
        gen_metrics = MetricsCalculator.compute_generation_metrics(answer, ground_truth)

        return {
            "id": item.get("id"),
            "question": question,
            "question_type": item.get("question_type", ""),
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "retrieved_contexts": contexts,
            "retrieved_payloads": payloads,
            "retrieved_law_ids": retrieved_law_ids,
            "law_id": item.get("law_id", {}),
            "is_graph": self.use_graph,
            "retrieval_latency_ms": round(retrieval_latency, 2),
            "generation_latency_ms": round(generation_latency, 2),
            "e2e_latency_ms": round(retrieval_latency + generation_latency, 2),
            **ret_metrics,
            **gen_metrics,
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "ragas_context_precision": None,
            "ragas_context_recall": None,
        }

    # ─────────────────────────────────────────────────────────────
    # Batch / All Evaluation
    # ─────────────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        batch_items: List[Dict[str, Any]],
        batch_idx: int = 1,
        total_batches: int = 1,
        on_case_completed: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Đánh giá 1 batch câu hỏi với multithreading để tăng tốc các request LLM/Retriever."""
        if not batch_items:
            return []

        import concurrent.futures

        workers = min(self.max_workers, len(batch_items))
        if workers > 1:
            results = [None] * len(batch_items)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Đảm bảo thứ tự câu hỏi không bị xáo trộn
                future_to_idx = {
                    executor.submit(self.evaluate_single, item): i
                    for i, item in enumerate(batch_items)
                }
                completed = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    res = future.result()
                    results[i] = res
                    if on_case_completed:
                        on_case_completed(res)
                    completed += 1
                    safe_print(
                        f"\r  [Batch {batch_idx}/{total_batches}] Hoàn thành: {completed}/{len(batch_items)} câu...",
                        end="",
                        flush=True,
                    )
            safe_print()
            return results
        else:
            results = []
            for idx, item in enumerate(batch_items, 1):
                if idx > 1:
                    time.sleep(1.5)  # Khoảng nghỉ an toàn giữa các câu hỏi để tránh rate limit
                qid = item.get("id", f"Q{idx}")
                q_preview = item.get("question", "")[:50]
                safe_print(
                    f"\r  [Batch {batch_idx}/{total_batches}] [{idx}/{len(batch_items)}] {qid}: {q_preview}...",
                    end="",
                    flush=True,
                )
                res = self.evaluate_single(item)
                results.append(res)
                if on_case_completed:
                    on_case_completed(res)
            safe_print()
            return results

    def evaluate_all(
        self,
        questions: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        on_case_completed: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Đánh giá toàn bộ danh sách câu hỏi theo từng batch có multithreading."""
        bs = batch_size or self.batch_size
        total = len(questions)
        batches = [questions[i : i + bs] for i in range(0, total, bs)]
        total_batches = len(batches)

        safe_print(f"  Tổng số: {total} câu hỏi | Chia thành: {total_batches} batch (batch_size={bs}, workers={self.max_workers})")

        all_results = []
        for b_idx, batch in enumerate(batches, 1):
            start_num = (b_idx - 1) * bs + 1
            end_num = min(b_idx * bs, total)
            safe_print(f"\n--- [Batch {b_idx}/{total_batches}] Đang xử lý câu hỏi {start_num} -> {end_num}/{total} ({len(batch)} câu) ---")
            batch_results = self.evaluate_batch(
                batch,
                batch_idx=b_idx,
                total_batches=total_batches,
                on_case_completed=on_case_completed,
            )
            all_results.extend(batch_results)
            safe_print(f"  ✓ Đã hoàn tất Batch {b_idx}/{total_batches}.")
            if b_idx < total_batches:
                time.sleep(2.0)  # Cooldown xả nghẽn API giữa các batch

        return all_results

    def free_gpu_resources(self) -> None:
        """
        Giải phóng toàn bộ tài nguyên GPU (VRAM) của các model thuộc session hiện tại:
        - Retriever models (DenseEmbedder, CrossEncoder Reranker, Contriever)
        - Local LLM generators (nếu có)
        - Processing pipelines (CRAG NLI, Prompt compressor, etc.)
        Được gọi tự động ngay khi kết thúc Phase 2 trước khi bước vào Phase 3 (RAGAS).
        """
        import gc
        import os

        pid = os.getpid()
        cfg = self.get_pipeline_metadata().get("configuration", {})
        sig = self.reporter.build_pipeline_signature(cfg) if hasattr(self, "reporter") else "default"

        safe_print(f"\n[Phase 2 -> Phase 3 | Session: '{sig}' | PID: {pid}] Đang giải phóng bộ nhớ GPU của session này...")
        freed_components = []

        # 1. Hủy Reranker & Embedder trong Retriever
        if hasattr(self, "retriever") and self.retriever is not None:
            if hasattr(self.retriever, "reranker") and self.retriever.reranker is not None:
                try:
                    del self.retriever.reranker
                    self.retriever.reranker = None
                    freed_components.append("Reranker (CrossEncoder)")
                except Exception as e:
                    logger.debug(f"[Cleanup] Lỗi khi hủy reranker: {e}")

            if hasattr(self.retriever, "embedder") and self.retriever.embedder is not None:
                try:
                    if hasattr(self.retriever.embedder, "model"):
                        del self.retriever.embedder.model
                        self.retriever.embedder.model = None
                    del self.retriever.embedder
                    self.retriever.embedder = None
                    freed_components.append("Embedder (Dense/Contriever)")
                except Exception as e:
                    logger.debug(f"[Cleanup] Lỗi khi hủy embedder: {e}")

            try:
                del self.retriever
                self.retriever = None
                freed_components.append("Retriever Pipeline")
            except Exception as e:
                logger.debug(f"[Cleanup] Lỗi khi hủy retriever: {e}")

        # 2. Hủy ProcessingManager
        if hasattr(self, "processing_manager") and self.processing_manager is not None:
            try:
                del self.processing_manager
                self.processing_manager = None
                freed_components.append("ProcessingManager")
            except Exception as e:
                logger.debug(f"[Cleanup] Lỗi khi hủy processing_manager: {e}")

        # 3. Hủy Local LLM nếu có
        if hasattr(self, "llm_manager") and self.llm_manager is not None:
            if getattr(self.llm_manager, "service", "") == "local":
                try:
                    if hasattr(self.llm_manager, "generator"):
                        del self.llm_manager.generator
                        self.llm_manager.generator = None
                    freed_components.append("Local LLM Generator")
                except Exception as e:
                    logger.debug(f"[Cleanup] Lỗi khi hủy local llm: {e}")

        if hasattr(self, "sub_llm_manager") and self.sub_llm_manager is not None:
            if getattr(self.sub_llm_manager, "service", "") == "local":
                try:
                    if hasattr(self.sub_llm_manager, "generator"):
                        del self.sub_llm_manager.generator
                        self.sub_llm_manager.generator = None
                    freed_components.append("Sub Local LLM Generator")
                except Exception as e:
                    logger.debug(f"[Cleanup] Lỗi khi hủy sub local llm: {e}")

        # 4. Thu gom rác Python
        gc.collect()

        # 5. Xả sạch bộ nhớ đệm PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)

                torch.cuda.empty_cache()

                allocated_after = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)
                vram_freed = reserved_before - reserved_after

                items_text = ", ".join(freed_components) if freed_components else "Các model session"
                safe_print(
                    f"  ✓ [PID {pid}] Đã giải phóng: {items_text}\n"
                    f"  ✓ [PID {pid}] VRAM giải phóng: {vram_freed:.1f} MB (Allocated: {allocated_after:.1f} MB, Reserved: {reserved_after:.1f} MB)"
                )
            else:
                items_text = ", ".join(freed_components) if freed_components else "Các model session"
                safe_print(f"  ✓ Đã thu dọn {items_text} (Đang chạy trên CPU).")
        except ImportError:
            safe_print("  ✓ Đã hoàn tất thu gom rác (PyTorch không khả dụng).")

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
        skip_base: bool = False,
        save: bool = True,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        ragas_max_workers: Optional[int] = None,
        ragas_batch_size: Optional[int] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ quy trình đánh giá hoàn chỉnh:
        - Tự động nhận diện phiên trước (Auto-Resume) nếu trùng metadata cấu hình.
        - Khởi tạo JSON ngay từ đầu và lưu theo phương thức append per-case.
        - Option --skip-base: Bỏ qua Phase 1 (Retrieval & Generation), giải phóng GPU ngay
          và nhảy thẳng sang Phase 3 (RAGAS Metrics) dựa trên các câu trả lời đã có.
        - Tính toán metrics và finalize báo cáo chuẩn hóa cho UI.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if max_workers is not None:
            self.max_workers = max_workers
        if ragas_max_workers is not None:
            self.ragas_max_workers = ragas_max_workers
        if ragas_batch_size is not None:
            self.ragas_batch_size = ragas_batch_size

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

        # 2. Xây dựng metadata cấu hình
        pipeline_meta = self.get_pipeline_metadata(
            eval_file=eval_file,
            limit=limit,
            random_sample=random_sample,
            seed=seed,
            skip_ragas=skip_ragas,
            skip_base=skip_base,
        )
        pipeline_meta["eval_dataset"] = {
            "law_name": dataset_meta.get("law_name", "Luật Đất đai 2024"),
            "law_number": dataset_meta.get("law_number", "31/2024/QH15"),
            "total_questions": len(questions),
            "version": dataset_meta.get("version", "2.1"),
        }

        # 3. Kiểm tra cơ chế Auto-Resume từ phiên trước
        resumable_session = None
        if resume and save:
            resumable_session = self.reporter.find_resumable_session(pipeline_meta["configuration"])

        completed_results = []
        pending_questions = questions
        report_path = None

        if resumable_session:
            report_path = resumable_session["report_path"]
            completed_results = resumable_session["completed_results"]
            completed_ids = resumable_session["completed_ids"]

            safe_print(f"\n[Auto-Resume] Phát hiện phiên chạy gần nhất TRÙNG KHỚP metadata ({resumable_session['matched_file']})!")
            safe_print(f"  → Đã hoàn thành: {len(completed_ids)}/{len(questions)} câu hỏi.")
        else:
            # Nếu không tìm thấy qua matching config, kiểm tra nếu eval_file chính là file kết quả đã có câu trả lời
            answers_found = [q for q in questions if q.get("generated_answer")]
            if answers_found:
                completed_results = answers_found
                completed_ids = {r.get("id") for r in completed_results if r.get("id")}
                report_path = Path(eval_file) if eval_file else self.reporter.get_latest_filepath(pipeline_meta["configuration"])
                safe_print(f"\n[File-Resume] Đã nhận diện file kết quả với {len(completed_results)} câu trả lời có sẵn.")

        # Xử lý đặc thù khi kích hoạt --skip-base
        if skip_base:
            if not completed_results:
                answers_in_questions = [q for q in questions if q.get("generated_answer")]
                if answers_in_questions:
                    completed_results = answers_in_questions
                    if not report_path and eval_file:
                        report_path = Path(eval_file)

            if not completed_results:
                safe_print("\n[Skip-Base Lỗi] Không tìm thấy câu trả lời nào đã có sẵn!")
                safe_print("  → Phiên chạy trùng khớp không có câu trả lời nào, hoặc file dữ liệu không chứa trường 'generated_answer'.")
                safe_print("  → Vui lòng chạy pipeline sinh câu trả lời trước, hoặc truyền file kết quả qua --eval-file.")
                return {}

            safe_print(f"\n[Skip-Base] Đã kích hoạt --skip-base:")
            safe_print(f"  ✓ Bỏ qua Phase 1 (Retrieval & Generation).")
            safe_print(f"  ✓ KHÔNG load bất kỳ mô hình GPU nào của session này (giữ nguyên GPU cho tiến trình khác).")
            safe_print(f"  ✓ Sử dụng {len(completed_results)} câu trả lời đã có để tiến hành đánh giá RAGAS metric.")
            pending_questions = []
        else:
            if resumable_session:
                pending_questions = [q for q in questions if q.get("id") not in completed_ids]
                if not pending_questions:
                    safe_print(f"  ✓ Toàn bộ {len(questions)} câu hỏi đã được thực hiện trước đó!")
                else:
                    safe_print(f"  → Tiếp tục thực hiện {len(pending_questions)} câu hỏi còn lại...\n")
            else:
                if save and not report_path:
                    report_path = self.reporter.init_streaming_session(
                        metadata=pipeline_meta,
                        total_questions=len(questions),
                    )

        # Tự động đảm bảo tất cả các case đã hoàn thành đều có đầy đủ tất cả các metric
        if completed_results:
            is_dirty = False
            for r in completed_results:
                if not self.use_graph and ("retrieval_hit" not in r or r.get("retrieval_hit") is None):
                    payloads = r.get("retrieved_payloads") or r.get("retrieved_law_ids") or []
                    ret_m = MetricsCalculator.compute_retrieval_metrics(payloads, r.get("law_id", {}))
                    r.update(ret_m)
                    is_dirty = True
                if "f1_score" not in r or r.get("f1_score") is None:
                    pred = r.get("generated_answer", "")
                    gt = r.get("ground_truth", "") or r.get("answer", "")
                    gen_m = MetricsCalculator.compute_generation_metrics(pred, gt)
                    r.update(gen_m)
                    is_dirty = True
                for mk in ["ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision", "ragas_context_recall"]:
                    if mk not in r:
                        r[mk] = None
                        is_dirty = True

            if is_dirty and save and report_path:
                safe_print("  → Đang bổ sung và chuẩn hóa toàn bộ metrics cho các case trước đó...")
                self.reporter.update_ragas_batch(report_path, completed_results, {})

        # 4. Định nghĩa callback append per-case
        def on_case_done(case_res):
            if save and report_path:
                self.reporter.append_case_result(
                    report_path=report_path,
                    case_result=case_res,
                    total_questions=len(questions),
                )

        # 5. Chạy các câu hỏi còn lại
        new_results = []
        if pending_questions:
            safe_print(f"\n[Phase 1] Đang chạy RAG pipeline ({self.retriever_mode}) cho {len(pending_questions)} câu hỏi...")
            new_results = self.evaluate_all(
                pending_questions,
                batch_size=self.batch_size,
                on_case_completed=on_case_done,
            )

        # Hợp nhất toàn bộ kết quả để tính toán metrics
        all_results = completed_results + new_results

        safe_print("\n[Phase 2] Tính toán các chỉ số cơ bản (Basic Metrics)...")
        basic_metrics = MetricsCalculator.aggregate(
            all_results,
            top_k=self.top_k,
            is_graph_mode=self.use_graph,
        )

        # ── KẾT THÚC PHASE 2: GIẢI PHÓNG VRAM / FREE GPU CÁC MODEL THUỘC SESSION (chỉ khi có chạy Phase 1) ──
        if not self.skip_base:
            self.free_gpu_resources()

        ragas_scores = {}
        if not skip_ragas:
            safe_print("\n[Phase 3] Tính toán RAGAS Metrics (LLM-as-a-judge)...")

            def on_ragas_batch_done(batch_items, cur_summary):
                if save and report_path:
                    self.reporter.update_ragas_batch(
                        report_path=report_path,
                        batch_results=batch_items,
                        current_ragas_summary=cur_summary,
                    )

            ragas_scores = self.ragas_judge.evaluate(
                all_results,
                batch_size=self.ragas_batch_size,
                max_workers=self.ragas_max_workers,
                on_batch_completed=on_ragas_batch_done,
            )
        else:
            safe_print("\n[Phase 3] Bỏ qua RAGAS Metrics (--skip-ragas).")

        # 6. In bảng kết quả
        self.reporter.print_summary_table(basic_metrics, ragas_scores, self.model_name)

        # 7. Clean payloads để lưu JSON an toàn
        clean_results = []
        for r in all_results:
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

        # 8. Lưu và finalize file JSON
        if save and report_path:
            self.reporter.finalize_unified_report(unified_output, report_path=report_path)

        return unified_output
