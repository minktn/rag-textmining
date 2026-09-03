"""
Advanced Retrieval Factory & Registry
=====================================
Quản lý đăng ký và khởi tạo động các chiến lược Advanced Retrieval:
Hỗ trợ mở rộng không giới hạn (RAG-Fusion, Self-RAG, ...) mà không cần sửa đổi Retriever pipeline.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Type

from .base import BaseAdvancedRetriever

logger = logging.getLogger(__name__)

_STRATEGY_REGISTRY: Dict[str, Type[BaseAdvancedRetriever]] = {}


def register_advanced(name: str):
    """Decorator để đăng ký một chiến lược Advanced Retriever mới."""
    def decorator(cls: Type[BaseAdvancedRetriever]):
        _STRATEGY_REGISTRY[name.lower().strip()] = cls
        return cls
    return decorator


def get_advanced_retriever(
    name: str,
    sub_llm_manager: Optional[Any] = None,
    **kwargs,
) -> BaseAdvancedRetriever:
    """
    Factory tạo đối tượng Advanced Retriever tương ứng với tên chiến lược.
    Tự động import package tương ứng trong `src.retriever.advanced.<name>` nếu chưa đăng ký.
    """
    name_clean = name.lower().strip()

    # 1. Thử dynamic import nếu chưa có trong registry
    if name_clean not in _STRATEGY_REGISTRY:
        # Thử import từ package src.retriever.advanced.<name_clean>
        for module_path in [
            f"src.retriever.advanced.{name_clean}",
            f"src.retriever.advanced.{name_clean}.retriever",
        ]:
            try:
                mod = importlib.import_module(module_path)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseAdvancedRetriever)
                        and attr is not BaseAdvancedRetriever
                    ):
                        _STRATEGY_REGISTRY[name_clean] = attr
                        break
                if name_clean in _STRATEGY_REGISTRY:
                    break
            except ImportError:
                continue

    # 2. Khởi tạo chiến lược
    if name_clean in _STRATEGY_REGISTRY:
        cls = _STRATEGY_REGISTRY[name_clean]
        return cls(sub_llm_manager=sub_llm_manager, **kwargs)

    raise ValueError(
        f"[Retriever] Phương thức advanced '{name}' chưa được triển khai hoặc không tìm thấy "
        f"class kế thừa BaseAdvancedRetriever trong src.retriever.advanced.{name_clean}."
    )
