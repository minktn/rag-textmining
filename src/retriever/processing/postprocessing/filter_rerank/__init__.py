"""
Filter-then-Rerank Postprocessing Module
=========================================
Triển khai phương pháp Filter-then-Rerank (EMNLP 2023):
- SLMFilter: Sàng lọc nhanh các mẫu dễ/không liên quan bằng SLM Local (settings.LOCAL_LLM).
- LLMReranker: Thẩm định chuyên sâu các mẫu khó (Hard Samples) bằng LLM Cloud (NVIDIA Nemotron).
- FilterReranker: Bộ điều phối toàn diện cho bước hậu xử lý truy xuất.
"""

from .filter import SLMFilter
from .reranker import LLMReranker
from .processor import FilterReranker

__all__ = ["SLMFilter", "LLMReranker", "FilterReranker"]
