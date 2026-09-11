"""
BM25 Indexing & Search Module
==============================
Triển khai thuật toán BM25Okapi chuẩn, hỗ trợ tách từ tiếng Việt qua Underthesea,
đóng gói và lưu trữ chỉ mục dưới định dạng pickle (bm25.pkl) cho Vector Store và Graph Database.
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def tokenize_vietnamese(text: str) -> List[str]:
    """Tách từ tiếng Việt chuẩn xác cho mô hình BM25.
    
    Ưu tiên sử dụng underthesea.word_tokenize nếu có, chuyển đổi các từ ghép thành dạng 'tu_ghep'.
    Fallback sang regex lọc từ nếu underthesea chưa được tải.
    """
    if not text or not isinstance(text, str):
        return []

    cleaned = text.lower().strip()
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(cleaned)
        # Thay thế khoảng trắng trong từ ghép thành dấu gạch dưới (ví dụ: 'đất đai' -> 'đất_đai')
        result = []
        for tok in tokens:
            tok = tok.strip()
            if tok and not re.match(r"^[\W_]+$", tok):
                result.append(tok.replace(" ", "_"))
        return result
    except Exception:
        # Fallback tách từ bằng regex đơn giản
        return re.findall(r"\w+", cleaned)


class BM25Okapi:
    """Cài đặt thuật toán BM25Okapi chuẩn (pure Python + NumPy) tối ưu hiệu năng và dễ serialize."""

    def __init__(
        self,
        corpus_tokens: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus_tokens)
        self.avgdl = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []

        if corpus_tokens:
            self._initialize(corpus_tokens)

    def _initialize(self, corpus_tokens: List[List[str]]):
        nd: Dict[str, int] = {}
        total_tokens = 0

        for tokens in corpus_tokens:
            self.doc_len.append(len(tokens))
            total_tokens += len(tokens)

            freqs: Dict[str, int] = {}
            for word in tokens:
                freqs[word] = freqs.get(word, 0) + 1
            self.doc_freqs.append(freqs)

            for word in freqs:
                nd[word] = nd.get(word, 0) + 1

        self.avgdl = (total_tokens / self.corpus_size) if self.corpus_size > 0 else 0.0

        # Tính chỉ số IDF theo công thức Okapi BM25
        idf_sum = 0.0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        average_idf = (idf_sum / len(self.idf)) if self.idf else 0.0
        eps = self.epsilon * average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """Tính điểm BM25 cho toàn bộ corpus dựa trên query tokens."""
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        doc_len = np.array(self.doc_len, dtype=np.float32)
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for q in query_tokens:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            q_freq = np.array([doc.get(q, 0) for doc in self.doc_freqs], dtype=np.float32)
            denominator = q_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
            scores += idf * (q_freq * (self.k1 + 1.0)) / np.where(denominator == 0, 1.0, denominator)

        return scores


class BM25Index:
    """Lớp quản lý chỉ mục BM25, cung cấp nạp, lưu (bm25.pkl) và truy vấn."""

    def __init__(
        self,
        bm25: BM25Okapi,
        chunks: List[Dict[str, Any]],
    ):
        self.bm25 = bm25
        self.chunks = chunks

    @classmethod
    def build(
        cls,
        chunks: List[Dict[str, Any]],
        text_key: str = "content",
    ) -> BM25Index:
        """Xây dựng chỉ mục BM25 từ danh sách chunk văn bản."""
        corpus_tokens = []
        for c in chunks:
            # Lấy text bao gồm cả metadata header nếu có
            text = c.get(text_key, "")
            metadata = c.get("metadata") or {}
            meta_str = " ".join(
                str(metadata.get(k, ""))
                for k in ("source", "chapter", "section", "article")
                if metadata.get(k)
            )
            full_text = f"{meta_str} {text}".strip() if meta_str else text
            corpus_tokens.append(tokenize_vietnamese(full_text))

        bm25_model = BM25Okapi(corpus_tokens)
        logger.info(f"[BM25] Đã xây dựng BM25Index với {len(chunks)} tài liệu.")
        return cls(bm25=bm25_model, chunks=chunks)

    def save(self, filepath: Union[str, Path]):
        """Lưu toàn bộ BM25 index ra file pkl."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bm25": self.bm25,
            "chunks": self.chunks,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[BM25] Đã lưu chỉ mục thành công vào: {path} ({len(self.chunks)} chunks)")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> BM25Index:
        """Nạp chỉ mục BM25 từ file pkl."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file BM25 index tại: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "bm25" in data and "chunks" in data:
            return cls(bm25=data["bm25"], chunks=data["chunks"])
        elif isinstance(data, cls):
            return data
        else:
            raise ValueError(f"Định dạng file {path} không hợp lệ cho BM25Index.")

    def search(
        self,
        query: str,
        top_k: int = 100,
        query_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tìm kiếm top_k tài liệu có điểm BM25 cao nhất."""
        if not self.chunks or self.bm25.corpus_size == 0:
            return []

        q_tokens = tokenize_vietnamese(query)
        scores = self.bm25.get_scores(q_tokens)

        # Áp dụng bộ lọc metadata nếu có
        ranked_indices = np.argsort(scores)[::-1]
        results = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]
            metadata = chunk.get("metadata") or {}

            # Kiểm tra query_filter
            if query_filter:
                match = True
                for k, v in query_filter.items():
                    if v is not None and metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            item = dict(chunk)
            item["bm25_score"] = float(scores[idx])
            item["bm25_rank"] = len(results) + 1
            item["corpus_index"] = int(idx)
            results.append(item)

            if len(results) >= top_k:
                break

        return results
