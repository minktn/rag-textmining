"""
Processing Architecture
=======================
Chỉ gồm đúng 2 class chung xử lý toàn bộ các phương thức:
- Preprocessor: Lớp chung điều phối tiền xử lý (preprocessing).
- Postprocessor: Lớp chung điều phối hậu xử lý (postprocessing).
"""

from .preprocessor import Preprocessor
from .postprocessor import Postprocessor

__all__ = [
    "Preprocessor",
    "Postprocessor",
]
