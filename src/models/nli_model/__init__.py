"""
NLI Model Module (BamiBERT ViLegalNLI)
=====================================
Cung cấp dịch vụ suy luận NLI cho hệ thống văn bản pháp luật tiếng Việt:
- Schemas: NLIItem, NLIRequest, NLIBatchRequest, NLIPrediction, NLIResponse, compute_sha256.
- FastAPI Server: Khởi chạy độc lập trên port 8001 qua `python -m src.models.nli_model.api`.
"""

from .schemas import (
    NLIItem,
    NLIRequest,
    NLIBatchRequest,
    NLIPrediction,
    NLIResponse,
    compute_sha256,
)

__all__ = [
    "NLIItem",
    "NLIRequest",
    "NLIBatchRequest",
    "NLIPrediction",
    "NLIResponse",
    "compute_sha256",
]
