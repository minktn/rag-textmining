"""
FastAPI NLI Provider API (Port 8001)
====================================
Dịch vụ suy luận NLI trên mô hình BamiBERT ViLegalNLI cho hệ thống văn bản pháp luật tiếng Việt.
Đây là dịch vụ microservice BẮT BUỘC phải khởi chạy để cung cấp NLI scoring cho pipeline CRAG.

CÁC ENDPOINTS:
--------------
- GET  /health         : Kiểm tra trạng thái dịch vụ và model.
- POST /predict        : Dự đoán 1 cặp (câu hỏi, văn bản luật).
- POST /predict_batch  : Dự đoán hàng loạt theo danh sách items (dành cho CRAG).
- POST /evaluate       : Đánh giá tập dữ liệu độc lập (Accuracy, Precision, Recall, F1).

CHÚ THÍCH LOG / AUDIT:
----------------------
Nếu request gửi lên có metadata `{"pipeline": "CRAG"}` hoặc `{"source": "CRAG"}`,
hệ thống sẽ tự động thêm tiền tố `[CRAG]` vào đầu các dòng log để phân biệt rõ ràng.
"""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings
from .logger import logger
from .schemas import (
    NLIBatchRequest,
    NLIItem,
    NLIPrediction,
    NLIRequest,
    NLIResponse,
    compute_sha256,
)


class NLIEngine:
    """Engine suy luận NLI nội bộ của FastAPI Server."""

    def __init__(
        self,
        model_dir: Optional[Path | str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent / "bamibert_vilegalnli"
        self.device = device or getattr(settings, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model_name = "BamiBERT-ViLegalNLI"

        logger.info(f"Đang nạp mô hình NLI từ: {self.model_dir} lên thiết bị: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        logger.info("Đã nạp thành công mô hình NLI.")

    def predict(self, question: str, context: str) -> Dict[str, Any]:
        """Dự đoán NLI cho 1 cặp (câu hỏi, ngữ cảnh)."""
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]

        prob_contra = round(float(probs[0].item()), 4)
        prob_entail = round(float(probs[1].item()), 4)
        pred_id = int(torch.argmax(probs).item())
        label = "ENTAILMENT/WIN" if pred_id == 1 else "CONTRADICTION/LOSE"
        confidence = prob_entail if pred_id == 1 else prob_contra

        return {
            "label": label,
            "label_id": pred_id,
            "confidence": confidence,
            "probabilities": {
                "CONTRADICTION/LOSE": prob_contra,
                "ENTAILMENT/WIN": prob_entail,
            },
        }

    def predict_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Dự đoán NLI theo lô cho danh sách các cặp (question, context)."""
        if not pairs:
            return []

        results: List[Dict[str, Any]] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            questions = [p[0] for p in batch]
            contexts = [p[1] for p in batch]

            inputs = self.tokenizer(
                questions,
                contexts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)

            for p_vec in probs:
                p_contra = round(float(p_vec[0].item()), 4)
                p_entail = round(float(p_vec[1].item()), 4)
                pred_id = int(torch.argmax(p_vec).item())
                label = "ENTAILMENT/WIN" if pred_id == 1 else "CONTRADICTION/LOSE"
                conf = p_entail if pred_id == 1 else p_contra

                results.append({
                    "label": label,
                    "label_id": pred_id,
                    "confidence": conf,
                    "probabilities": {
                        "CONTRADICTION/LOSE": prob_contra,
                        "ENTAILMENT/WIN": prob_entail,
                    },
                })
        return results


# Global engine instance
nli_engine: Optional[NLIEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo mô hình NLI vào bộ nhớ GPU/CPU khi server khởi chạy."""
    global nli_engine
    logger.info("🚀 Đang khởi động FastAPI NLI Provider (Port 8001)...")
    nli_engine = NLIEngine()
    logger.info(f"🎯 Mô hình {nli_engine.model_name} đã sẵn sàng phục vụ trên: {nli_engine.device}")
    yield
    logger.info("🛑 FastAPI NLI Provider đã dừng.")


app = FastAPI(
    title="Vietnamese Legal NLI Provider API",
    description="API suy luận NLI (Natural Language Inference) cho văn bản pháp luật tiếng Việt",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_log_prefix(metadata: Optional[Dict[str, Any]]) -> str:
    """Xác định tiền tố log: trả về '[CRAG] ' nếu request thuộc pipeline CRAG."""
    if metadata and (metadata.get("pipeline") == "CRAG" or metadata.get("source") == "CRAG"):
        return "[CRAG] "
    return ""


# ==============================================================================
# 1. HEALTH CHECK
# ==============================================================================

@app.get("/health", summary="Kiểm tra trạng thái dịch vụ và mô hình NLI")
def health_check():
    global nli_engine
    if nli_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLI Engine chưa sẵn sàng.",
        )
    return {
        "status": "healthy",
        "service": "VietnameseLegalNLIProvider",
        "model_name": nli_engine.model_name,
        "local_model_dir": str(nli_engine.model_dir),
        "device": str(nli_engine.device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ==============================================================================
# 2. INFERENCE ENDPOINTS (SINGLE & BATCH)
# ==============================================================================

@app.post("/predict", response_model=NLIResponse, summary="Dự đoán NLI đơn lẻ (1 cặp câu)")
def predict_single(request: NLIRequest):
    """
    Input: JSON chứa `specific_question` và `legal_document`.
    Output: JSON chứa kết quả phân loại, xác suất, metadata, latency và SHA-256 hash.
    """
    global nli_engine
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    input_sha256 = compute_sha256(request.model_dump_json())

    item = NLIItem(
        specific_question=request.specific_question,
        legal_document=request.legal_document,
        id=request.id,
        metadata=request.metadata,
    )
    item_sha256 = item.get_sha256()

    raw_res = nli_engine.predict(
        question=request.specific_question,
        context=request.legal_document,
    )

    prediction = NLIPrediction(
        id=request.id,
        label=str(raw_res["label"]),
        label_id=int(raw_res["label_id"]),
        confidence=float(raw_res["confidence"]),
        probabilities=raw_res["probabilities"],
        item_sha256=item_sha256,
        metadata=request.metadata,
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    log_prefix = _get_log_prefix(request.metadata)

    logger.info(
        f"{log_prefix}[PREDICT] req_id={request_id} label={prediction.label} "
        f"conf={prediction.confidence} latency={latency_ms}ms input_sha256={input_sha256[:12]}..."
    )

    return NLIResponse(
        request_id=request_id,
        timestamp=timestamp,
        model_name=nli_engine.model_name,
        device=str(nli_engine.device),
        latency_ms=latency_ms,
        input_sha256=input_sha256,
        prediction=prediction,
        total_items=1,
        session_id=request.session_id,
        metadata=request.metadata,
    )


@app.post("/predict_batch", response_model=NLIResponse, summary="Dự đoán NLI theo lô (Batch)")
def predict_batch(request: NLIBatchRequest):
    """
    Input: JSON chứa `items: [{"specific_question": "...", "legal_document": "..."}, ...]`.
    Output: JSON danh sách kết quả, độ trễ và metadata tổng thể.
    """
    global nli_engine
    if not request.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Danh sách 'items' không được để trống.",
        )

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    input_sha256 = compute_sha256(request.model_dump_json())

    pairs = [(item.specific_question, item.legal_document) for item in request.items]
    raw_results = nli_engine.predict_batch(pairs)

    predictions: List[NLIPrediction] = []
    for item, raw_res in zip(request.items, raw_results):
        predictions.append(
            NLIPrediction(
                id=item.id,
                label=str(raw_res["label"]),
                label_id=int(raw_res["label_id"]),
                confidence=float(raw_res["confidence"]),
                probabilities=raw_res["probabilities"],
                item_sha256=item.get_sha256(),
                metadata=item.metadata,
            )
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    log_prefix = _get_log_prefix(request.metadata)

    logger.info(
        f"{log_prefix}[PREDICT_BATCH] req_id={request_id} count={len(predictions)} "
        f"latency={latency_ms}ms input_sha256={input_sha256[:12]}..."
    )

    return NLIResponse(
        request_id=request_id,
        timestamp=timestamp,
        model_name=nli_engine.model_name,
        device=str(nli_engine.device),
        latency_ms=latency_ms,
        input_sha256=input_sha256,
        predictions=predictions,
        total_items=len(predictions),
        session_id=request.session_id,
        metadata=request.metadata,
    )


# ==============================================================================
# 3. BENCHMARK / DATASET EVALUATION ENDPOINT
# ==============================================================================

@app.post("/evaluate", summary="Đánh giá tập dữ liệu độc lập (Accuracy, Precision, Recall, F1)")
def evaluate_dataset(payload: List[Dict[str, Any]]):
    """
    Input: Danh sách mẫu dữ liệu test từ evaluate.ipynb / Parquet.
    Output: JSON gồm metrics và dự đoán chi tiết.
    """
    global nli_engine
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tập dữ liệu không được để trống.",
        )

    start_time = time.perf_counter()
    pairs = []
    y_true = []
    has_labels = True

    for sample in payload:
        q = sample.get("specific_question") or sample.get("question", "")
        ctx = sample.get("legal_document") or sample.get("context", "")
        pairs.append((q, ctx))

        if "answer" in sample:
            ans = sample["answer"]
            y_true.append(1 if ans == 0 else 0) if isinstance(ans, int) else y_true.append(int(ans))
        elif "label" in sample:
            y_true.append(int(sample["label"]))
        else:
            has_labels = False

    raw_results = nli_engine.predict_batch(pairs)
    y_pred = [r["label_id"] for r in raw_results]

    metrics = None
    if has_labels and len(y_true) == len(y_pred):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        metrics = {
            "total_samples": len(y_true),
            "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "Precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "F1-Score": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "Macro F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        }

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(f"[EVALUATE] total_samples={len(pairs)} latency={latency_ms}ms metrics={metrics}")

    return {
        "status": "success",
        "total_samples": len(pairs),
        "latency_ms": latency_ms,
        "metrics": metrics,
        "predictions": raw_results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.models.nli_model.api:app", host="0.0.0.0", port=8001, reload=True)
