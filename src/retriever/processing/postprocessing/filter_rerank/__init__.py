"""
Filter-then-Rerank Postprocessing Module
=========================================
Implements the two-tier Filter-then-Rerank framework (EMNLP 2023):
- SLMFilter: High-throughput preliminary candidate filtering via local SLM logits.
- LLMReranker: Selective multi-choice reasoning for hard/ambiguous samples via Cloud LLM.
- FilterReranker: Unified pipeline coordinator for retriever postprocessing.
"""

from .filter import SLMFilter
from .processor import FilterReranker
from .reranker import LLMReranker

__all__ = ["SLMFilter", "LLMReranker", "FilterReranker"]
