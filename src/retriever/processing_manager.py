"""
Processing Manager
==================
Quản lý cấu hình pipeline xử lý cho hệ thống RAG:
- preprocessing: Các bước tiền xử lý truy vấn (query_rewriter luôn chạy riêng, không cần khai báo)
- postprocessing: Các bước hậu xử lý kết quả truy xuất
- advanced: Phương thức truy xuất nâng cao — thay thế toàn bộ pipeline (ưu tiên cao nhất)

Quy tắc xác thực:
- Nếu `advanced` được chỉ định → bỏ qua toàn bộ preprocessing và postprocessing.
- Đối với preprocessing/postprocessing: nếu len > 1, tối đa 1 phương thức KHÔNG nằm trong
  danh sách ADAPTABLE tương ứng (các phương thức không adaptable không thể kết hợp cùng nhau).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.config import settings

logger = logging.getLogger(__name__)

# Directories for scanning available methods
_RETRIEVER_DIR = Path(__file__).resolve().parent
_PREPROCESSING_DIR = _RETRIEVER_DIR / "processing" / "preprocessing"
_POSTPROCESSING_DIR = _RETRIEVER_DIR / "processing" / "postprocessing"
_ADVANCED_DIR = _RETRIEVER_DIR / "advanced"


def _scan_folder_names(directory: Path) -> List[str]:
	"""Scan child folder names in a directory (excluding __pycache__ etc.)."""
	if not directory.exists() or not directory.is_dir():
		return []
	return sorted([
		d.name for d in directory.iterdir()
		if d.is_dir() and not d.name.startswith("__")
	])


class ProcessingManager:
	"""Quản lý và xác thực cấu hình preprocessing, postprocessing, và advanced retrieval.

	Parameters
	----------
	preprocessing : List[str] | None
		Danh sách phương thức tiền xử lý. Keywords = tên thư mục con trong preprocessing/.
	postprocessing : List[str] | None
		Danh sách phương thức hậu xử lý. Keywords = tên thư mục con trong postprocessing/.
	advanced : str | None
		Phương thức truy xuất nâng cao. Ưu tiên cao nhất — nếu được chỉ định,
		preprocessing và postprocessing sẽ bị bỏ qua.
		Keywords = tên thư mục con trong advanced/.
	"""

	def __init__(
		self,
		preprocessing: Optional[List[str]] = None,
		postprocessing: Optional[List[str]] = None,
		advanced: Optional[str] = None,
	):
		self.advanced = (advanced or "").strip()

		# Advanced has highest priority — reject pre/postprocessing when set
		if self.advanced:
			self.preprocessing: List[str] = []
			self.postprocessing: List[str] = []
			logger.info(
				f"[ProcessingManager] Advanced mode: '{self.advanced}'. "
				f"Preprocessing and postprocessing are disabled."
			)
		else:
			self.preprocessing = list(preprocessing) if preprocessing else []
			self.postprocessing = list(postprocessing) if postprocessing else []

		self._validate()

		# Khởi tạo 2 class chung xử lý pre/post processing
		from src.retriever.processing import Postprocessor, Preprocessor
		self.preprocessor = Preprocessor(methods=self.preprocessing)
		self.postprocessor = Postprocessor(methods=self.postprocessing)

	@classmethod
	def from_settings(cls) -> ProcessingManager:
		"""Khởi tạo ProcessingManager mặc định cho Baseline:
		Thuần túy Standard Pipeline (advanced=None, preprocessing=[], postprocessing=[]).
		"""
		return cls(
			preprocessing=[],
			postprocessing=[],
			advanced=None,
		)

	# ── Keyword Discovery (scan folder names) ─────────────────

	@staticmethod
	def preprocess_keywords() -> List[str]:
		"""Scan preprocessing/ child folder names → available preprocessing methods."""
		return _scan_folder_names(_PREPROCESSING_DIR)

	@staticmethod
	def postprocess_keywords() -> List[str]:
		"""Scan postprocessing/ child folder names → available postprocessing methods."""
		return _scan_folder_names(_POSTPROCESSING_DIR)

	@staticmethod
	def advanced_keywords() -> List[str]:
		"""Scan advanced/ child folder names → available advanced retrieval methods."""
		return _scan_folder_names(_ADVANCED_DIR)

	# ── Validation ─────────────────────────────────────────────

	def _validate(self):
		"""Xác thực toàn bộ cấu hình pipeline."""
		if self.advanced:
			self._validate_advanced()
			return

		if self.preprocessing:
			self._validate_method_list(
				items=self.preprocessing,
				available=self.preprocess_keywords(),
				adaptable=list(getattr(settings, "ADAPTABLE_PREPROCESS", [])),
				label="preprocessing",
			)

		if self.postprocessing:
			self._validate_method_list(
				items=self.postprocessing,
				available=self.postprocess_keywords(),
				adaptable=list(getattr(settings, "ADAPTABLE_POSTPROCESS", [])),
				label="postprocessing",
			)

	def _validate_advanced(self):
		"""Xác thực advanced method tồn tại."""
		available = self.advanced_keywords()
		if self.advanced not in available:
			raise ValueError(
				f"[ProcessingManager] Advanced retrieval '{self.advanced}' không hợp lệ. "
				f"Các phương thức khả dụng: {available}"
			)

	@staticmethod
	def _validate_method_list(
		items: List[str],
		available: List[str],
		adaptable: List[str],
		label: str,
	):
		"""Xác thực danh sách phương thức pre/postprocessing."""
		for item in items:
			if item not in available:
				raise ValueError(
					f"[ProcessingManager] {label} '{item}' không hợp lệ. "
					f"Các phương thức khả dụng: {available}"
				)

		if len(items) > 1:
			non_adaptable = [item for item in items if item not in adaptable]
			if len(non_adaptable) > 1:
				raise ValueError(
					f"[ProcessingManager] Cấu hình {label} không hợp lệ: "
					f"có {len(non_adaptable)} phương thức không nằm trong danh sách adaptable ({non_adaptable}). "
					f"Chỉ cho phép kết hợp tối đa 1 phương thức không nằm trong danh sách adaptable ({adaptable})."
				)

	# ── Execution (Uỷ quyền trực tiếp cho Preprocessor & Postprocessor) ─

	def apply_preprocessing(
		self,
		query: str,
		retriever: Optional[Any] = None,
	) -> str:
		"""Thực thi tiền xử lý qua lớp chung Preprocessor."""
		return self.preprocessor.process(query=query, retriever=retriever)

	def apply_postprocessing(
		self,
		query: str,
		chunks: List[Dict[str, Any]],
		retriever: Optional[Any] = None,
	) -> List[Dict[str, Any]]:
		"""Thực thi hậu xử lý qua lớp chung Postprocessor."""
		return self.postprocessor.process(query=query, chunks=chunks, retriever=retriever)

	# ── Utility ────────────────────────────────────────────────

	@property
	def is_standard(self) -> bool:
		"""True nếu pipeline không sử dụng advanced, preprocessing, hoặc postprocessing."""
		return not self.advanced and not self.preprocessing and not self.postprocessing

	def __repr__(self) -> str:
		if self.advanced:
			return f"ProcessingManager(advanced='{self.advanced}')"
		parts = []
		if self.preprocessing:
			parts.append(f"preprocessing={self.preprocessing}")
		if self.postprocessing:
			parts.append(f"postprocessing={self.postprocessing}")
		if not parts:
			return "ProcessingManager(standard)"
		return f"ProcessingManager({', '.join(parts)})"