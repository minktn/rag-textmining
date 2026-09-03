"""
FastAPI Server for RAG Application & Evaluation Playground (Port 8002)
======================================================================
Cung cấp REST API cho:
1. Chatbot RAG Inference: POST /query
2. Trạng thái hệ thống: GET /health
3. Thông tin tập dữ liệu đánh giá: GET /eval/info
4. Chạy trực tiếp Evaluation: POST /eval/run
5. Tải báo cáo đánh giá mới nhất: GET /eval/latest
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.retriever import ProcessingManager, Retriever
from src.generation import LLMManager, SubLLMManager
from src.evaluation import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="VietLegal RAG API Server",
    description="Backend API phục vụ Chatbot hỏi đáp pháp luật và Evaluation Playground",
    version="1.0.0",
)

# Kích hoạt CORS cho React UI (Vite dev server cổng 5173 / preview)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request & Response Models ───────────────────────

class QueryRequest(BaseModel):
    query: str
    rag: bool = True
    top_k: int = 5
    database: str = "base"  # "base" (BGE-M3), "contriever", "graph"
    advanced: Optional[str] = None  # "rag_fusion"
    preprocessing: Optional[List[str]] = None  # ["hyde"]
    postprocessing: Optional[List[str]] = None  # ["filter_rerank", "crag", "prompt_compression"]
    llm_service: Optional[str] = None
    sub_llm_service: Optional[str] = None
    session_id: Optional[str] = None


class RetrievedChunkResponse(BaseModel):
    id: Optional[str] = None
    text: str
    score: Optional[float] = None
    source_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    request_id: str
    timestamp: str
    query: str
    rewritten_query: Optional[str] = None
    rag_used: bool
    retrieval_mode: str
    answer: str
    retrieved_chunks: List[RetrievedChunkResponse]
    total_chunks: int
    latency_ms: float
    input_sha256: str
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EvalRunRequest(BaseModel):
    limit: int = Field(default=1, ge=1, description="Số lượng câu hỏi cần đánh giá")
    random_sample: bool = True
    random_seed: int = 42
    retriever_mode: str = "base"  # "base" | "contriever" | "graph"
    advanced: Optional[str] = None
    preprocessing: Optional[List[str]] = None
    postprocessing: Optional[List[str]] = None
    llm_service: Optional[str] = None
    sub_llm_service: Optional[str] = None
    ragas_service: Optional[str] = None
    skip_ragas: bool = True
    top_k: int = 5


# ── Health Check ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "timestamp": time.time(),
        "device": getattr(settings, "DEVICE", "cpu"),
        "prefer_local_storage": getattr(settings, "PREFER_LOCAL_STORAGE", True),
    }


# ── Chatbot Inference ────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    input_sha256 = hashlib.sha256(req.query.encode("utf-8")).hexdigest()

    llm_service = req.llm_service or getattr(settings, "LLM_SERVICE", "nvidia")
    sub_llm_service = req.sub_llm_service or getattr(settings, "SUB_LLM_SERVICE", "nvidia")
    llm = LLMManager(service=llm_service)

    # 1. Chế độ Direct LLM (Không dùng RAG)
    if not req.rag:
        try:
            answer = llm.generate_response(req.query) or ""
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return QueryResponse(
                request_id=request_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                query=req.query,
                rewritten_query=None,
                rag_used=False,
                retrieval_mode="direct_llm",
                answer=answer,
                retrieved_chunks=[],
                total_chunks=0,
                latency_ms=latency_ms,
                input_sha256=input_sha256,
                session_id=req.session_id,
            )
        except Exception as e:
            logger.error(f"[Query] Direct LLM error: {e}")
            raise HTTPException(status_code=500, detail=f"Direct LLM error: {e}")

    # 2. Chế độ RAG (Vector DB / Contriever / Graph DB)
    try:
        pm = ProcessingManager(
            preprocessing=req.preprocessing,
            postprocessing=req.postprocessing,
            advanced=req.advanced,
        )

        if req.database in ("graph", "neo4j", "graphrag"):
            dense_model = settings.EMBEDDING_MODEL
            coll_name = "graph"
        elif req.database == "contriever":
            dense_model = settings.CONTRIEVER_MODEL
            coll_name = settings.CONTRIEVER_COLLECTION_NAME
        else:
            dense_model = settings.EMBEDDING_MODEL
            coll_name = settings.COLLECTION_NAME

        retriever = Retriever(
            dense_model_name=dense_model,
            collection_name=coll_name,
            sub_llm_manager=SubLLMManager(service=sub_llm_service),
            processing_manager=pm,
            rerank_limit=req.top_k,
        )

        retrieval_res = retriever.retrieve(req.query)
        chunks = retrieval_res.get("chunks") or retrieval_res.get("selected_chunks") or []
        if req.top_k and len(chunks) > req.top_k:
            chunks = chunks[: req.top_k]

        # Sinh câu trả lời với context
        prompt = llm.construct_prompt(req.query, docs=chunks)
        answer = llm.generate_response(prompt) or ""

        retrieved_chunk_objs = []
        for c in chunks:
            retrieved_chunk_objs.append(
                RetrievedChunkResponse(
                    id=str(c.get("id", "")),
                    text=c.get("content", ""),
                    score=c.get("rerank_score") or c.get("dense_score"),
                    source_type=c.get("source", "dense"),
                    metadata=c.get("metadata", {}),
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return QueryResponse(
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            query=req.query,
            rewritten_query=retrieval_res.get("normalized_query"),
            rag_used=True,
            retrieval_mode=req.database,
            answer=answer,
            retrieved_chunks=retrieved_chunk_objs,
            total_chunks=len(retrieved_chunk_objs),
            latency_ms=latency_ms,
            input_sha256=input_sha256,
            session_id=req.session_id,
            metadata={
                "advanced": pm.advanced or None,
                "preprocessing": pm.preprocessing,
                "postprocessing": pm.postprocessing,
                "database": req.database,
            },
        )
    except Exception as e:
        logger.error(f"[Query] Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Evaluation Endpoints ─────────────────────────────────────

@app.get("/eval/info")
def get_eval_info():
    """Lấy thông tin tập dữ liệu đánh giá và các cấu hình khả dụng."""
    eval_file = settings.EVAL_DATA_DIR / "eval_landlaw_2024.json"
    total_questions = 0
    law_name = "Luật Đất đai 2024"
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                items = d.get("data") or d.get("questions") or []
                total_questions = len(items)
                meta = d.get("metadata") or {}
                law_name = meta.get("law", law_name)
        except Exception as e:
            logger.warning(f"Không thể đọc file eval: {e}")

    return {
        "dataset_name": "eval_landlaw_2024.json",
        "law": law_name,
        "total_questions": total_questions,
        "default_seed": 42,
        "available_options": {
            "database": [
                {"id": "base", "name": "Vector DB (BGE-M3)", "desc": "Dense search với BGE-M3 & Cross-Encoder"},
                {"id": "contriever", "name": "Contriever", "desc": "Dense search với mContriever"},
                {"id": "graph", "name": "Graph Database", "desc": "Microsoft GraphRAG trên Knowledge Graph"},
            ],
            "advanced": [
                {"id": "rag_fusion", "name": "RAG-Fusion", "desc": "Multi-query expansion qua Sub-LLM + Reciprocal Rank Fusion"},
            ],
            "preprocessing": [
                {"id": "hyde", "name": "HyDE", "desc": "Hypothetical Document Embeddings"},
            ],
            "postprocessing": [
                {"id": "filter_rerank", "name": "Filter-then-Rerank", "desc": "Local SLM Filter (1.5B) -> NVIDIA LLM Reranker"},
                {"id": "crag", "name": "Corrective RAG (CRAG)", "desc": "Kiểm chứng tài liệu qua NLI"},
                {"id": "prompt_compression", "name": "Prompt Compression", "desc": "Nén ngữ cảnh LongLLMLingua"},
            ],
            "llm_services": ["nvidia", "groq", "google", "local"],
        },
    }


@app.post("/eval/run")
def run_evaluation(req: EvalRunRequest):
    """Thực thi pipeline đánh giá và trả về kết quả JSON trực tiếp cho UI."""
    eval_file = settings.EVAL_DATA_DIR / "eval_landlaw_2024.json"
    if not eval_file.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy tập dữ liệu đánh giá.")

    # Đọc số lượng mẫu tối đa
    with open(eval_file, "r", encoding="utf-8") as f:
        d = json.load(f)
        total_questions = len(d.get("data") or d.get("questions") or [])

    if req.limit <= 0:
        raise HTTPException(status_code=400, detail="Số lượng mẫu thử phải lớn hơn 0.")
    if req.limit > total_questions:
        raise HTTPException(
            status_code=400,
            detail=f"Số lượng mẫu thử ({req.limit}) vượt quá số lượng tối đa trong tập dữ liệu ({total_questions}).",
        )

    logger.info(
        f"[Eval API] Bắt đầu đánh giá: N={req.limit}, random={req.random_sample} (seed={req.random_seed}), "
        f"mode={req.retriever_mode}, adv={req.advanced}, pre={req.preprocessing}, post={req.postprocessing}"
    )

    try:
        evaluator = RAGEvaluator(
            retriever_mode=req.retriever_mode,
            llm_service=req.llm_service or getattr(settings, "LLM_SERVICE", "nvidia"),
            sub_llm_service=req.sub_llm_service or getattr(settings, "SUB_LLM_SERVICE", "nvidia"),
            advanced=req.advanced,
            preprocessing=req.preprocessing,
            postprocessing=req.postprocessing,
            ragas_service=req.ragas_service or getattr(settings, "RAGAS_SERVICE", "nvidia"),
            top_k=req.top_k,
        )

        results = evaluator.run(
            eval_file=eval_file,
            limit=req.limit,
            random_sample=req.random_sample,
            seed=req.random_seed,
            skip_ragas=req.skip_ragas,
            save=True,
        )

        # Đọc báo cáo mới nhất từ db/results/eval_latest.json
        latest_file = settings.EVAL_RESULTS_DIR / "eval_latest.json"
        if latest_file.exists():
            with open(latest_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            return report_data

        return results
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[Eval API] Lỗi khi chạy đánh giá: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Lỗi đánh giá: {str(e)}\n{tb}")


@app.get("/eval/latest")
def get_latest_eval():
    """Lấy báo cáo đánh giá đã lưu mới nhất."""
    latest_file = settings.EVAL_RESULTS_DIR / "eval_latest.json"
    if not latest_file.exists():
        raise HTTPException(status_code=404, detail="Chưa có báo cáo đánh giá nào được lưu.")
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
