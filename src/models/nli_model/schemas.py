"""
NLI Data Schemas & Models
=========================
Định nghĩa Pydantic Schemas phục vụ dịch vụ NLI Provider (FastAPI Port 8001).
Hỗ trợ metadata phục vụ phân loại luồng gọi (ví dụ: metadata={'pipeline': 'CRAG'}).
"""

import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


def compute_sha256(text: str) -> str:
    """Tính toán SHA-256 hash của một chuỗi văn bản phục vụ audit log và tracking."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NLIItem(BaseModel):
    """Một cặp câu hỏi/nhận định và tài liệu/ngữ cảnh pháp lý."""
    model_config = ConfigDict(populate_by_name=True)

    specific_question: str = Field(
        ...,
        alias="question",
        description="Câu hỏi hoặc nhận định cần kiểm chứng (Hypothesis/Question)",
    )
    legal_document: str = Field(
        ...,
        alias="context",
        description="Tài liệu luật hoặc ngữ cảnh đối chiếu (Premise/Context)",
    )
    id: Optional[str] = Field(
        default=None,
        description="ID định danh mẫu dữ liệu (tùy chọn)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata tùy chỉnh (ví dụ: pipeline='CRAG', session_id, turn_id)",
    )

    def get_sha256(self) -> str:
        """Tạo sha256 duy nhất từ cặp (specific_question, legal_document)."""
        combined = f"{self.specific_question.strip()}\n---\n{self.legal_document.strip()}"
        return compute_sha256(combined)


class NLIRequest(BaseModel):
    """Schema input cho dự đoán đơn lẻ."""
    model_config = ConfigDict(populate_by_name=True)

    specific_question: str = Field(
        ...,
        alias="question",
        description="Câu hỏi hoặc nhận định cần kiểm chứng",
    )
    legal_document: str = Field(
        ...,
        alias="context",
        description="Tài liệu pháp luật đối chiếu",
    )
    id: Optional[str] = Field(default=None, description="ID của request hoặc item")
    session_id: Optional[str] = Field(default=None, description="ID phiên làm việc")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata phục vụ monitoring & phân định luồng (ví dụ: {'pipeline': 'CRAG'})",
    )


class NLIBatchRequest(BaseModel):
    """Schema input cho dự đoán hàng loạt (Batch)."""
    items: List[NLIItem] = Field(..., description="Danh sách các cặp câu hỏi - tài liệu luật")
    session_id: Optional[str] = Field(default=None, description="ID phiên làm việc / đánh giá")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata chung cho toàn bộ batch (ví dụ: {'pipeline': 'CRAG'})",
    )


class NLIPrediction(BaseModel):
    """Kết quả phân loại NLI chi tiết cho 1 mẫu."""
    id: Optional[str] = Field(default=None, description="ID mẫu dữ liệu")
    label: str = Field(..., description="Nhãn dự đoán: 'ENTAILMENT/WIN' hoặc 'CONTRADICTION/LOSE'")
    label_id: int = Field(..., description="Mã số nhãn: 1 (Entailment) hoặc 0 (Contradiction)")
    confidence: float = Field(..., description="Độ tin cậy của nhãn dự đoán (0.0 - 1.0)")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Phân phối xác suất cho tất cả các nhãn: {'CONTRADICTION/LOSE': float, 'ENTAILMENT/WIN': float}",
    )
    item_sha256: str = Field(..., description="Mã hash SHA-256 của cặp câu hỏi - tài liệu")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata kèm theo nếu có")


class NLIResponse(BaseModel):
    """Schema output JSON trả về kèm metadata phục vụ monitoring & audit log."""
    request_id: str = Field(..., description="UUID duy nhất của lượt gọi API")
    timestamp: str = Field(..., description="Thời gian thực thi (ISO 8601 UTC)")
    model_name: str = Field(..., description="Tên mô hình NLI đang phục vụ")
    device: str = Field(..., description="Thiết bị thực thi (cuda / cpu)")
    latency_ms: float = Field(..., description="Thời gian xử lý tính bằng mili-giây (ms)")
    input_sha256: str = Field(..., description="Mã hash SHA-256 của toàn bộ payload đầu vào")
    prediction: Optional[NLIPrediction] = Field(default=None, description="Kết quả cho request đơn")
    predictions: Optional[List[NLIPrediction]] = Field(default=None, description="Kết quả cho request batch")
    total_items: int = Field(default=1, description="Tổng số item được xử lý")
    session_id: Optional[str] = Field(default=None, description="ID phiên làm việc")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata tổng hợp")
