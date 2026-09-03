"""
Data Ingestion & Chunking Module
================================
Cung cấp pipeline chính thức để chunking và nạp dữ liệu vào các hệ thống lưu trữ:
- BaseStore: BGE-M3 Dense + BM25 Sparse (Standard & Late Chunking)
- ContrieverStore: mContriever Dense
- Neo4jGraphStore: Knowledge Graph từ GraphRAG Parquet & LanceDB

Có thể gọi qua hàm (programmatic API) hoặc chạy trực tiếp từ CLI:
    python -m src.database.ingest --store base
    python -m src.database.ingest --chunk-only
    python -m src.database.ingest --status
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the workspace root is in sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import settings
from src.database.store_manager import StoreManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline điều phối toàn bộ quy trình chunking và nạp vector/graph database."""

    def __init__(self, manager: Optional[StoreManager] = None):
        self.manager = manager or StoreManager()

    def run_chunking(
        self,
        input_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Tách văn bản luật thô (.md) thành danh sách chunk có cấu trúc (.json)."""
        logger.info("[Ingest] Bắt đầu chunking văn bản luật...")
        chunks = self.manager.chunk(input_file=input_file, output_file=output_file)
        logger.info(f"[Ingest] Hoàn tất chunking: {len(chunks)} chunks đã được tạo.")
        return chunks

    def run_ingest(
        self,
        store_type: str = "base",
        late_chunking: bool = False,
        sync_to_cloud: bool = True,
        rechunk: bool = False,
        build_graph: bool = False,
        input_file: Optional[Path] = None,
        chunks_file: Optional[Path] = None,
    ):
        """Thực hiện quy trình nạp dữ liệu hoàn chỉnh cho store chỉ định."""
        logger.info(
            f"[Ingest] Bắt đầu nạp dữ liệu: store='{store_type}', "
            f"late={late_chunking}, sync_cloud={sync_to_cloud}, rechunk={rechunk}, build_graph={build_graph}"
        )
        self.manager.ingest(
            store_type=store_type,
            late_chunking=late_chunking,
            sync_to_cloud=sync_to_cloud,
            rechunk=rechunk,
            build_graph=build_graph,
            input_file=input_file,
            chunks_file=chunks_file,
        )
        logger.info("[Ingest] Nạp dữ liệu hoàn tất thành công.")

    def get_status(self) -> Dict[str, Any]:
        """Xem trạng thái dữ liệu hiện có ở local và cloud."""
        return self.manager.status()


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Legal RAG - Ingestion Pipeline")
    parser.add_argument(
        "--store",
        type=str,
        default="base",
        choices=["base", "contriever", "graph", "all"],
        help="Mục tiêu lưu trữ ('base', 'contriever', 'graph', 'all')",
    )
    parser.add_argument(
        "--late",
        action="store_true",
        help="Sử dụng Late Chunking cho Base Store (bảo toàn ngữ cảnh văn bản dài)",
    )
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="Chỉ thực hiện tách chunk từ file Markdown sang JSON, không sinh vector embedding",
    )
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="Ép buộc chunk lại từ đầu kể cả khi file JSON chunks đã tồn tại",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Kích hoạt 'graphrag index' để trích xuất thực thể, quan hệ và xây dựng đồ thị từ file gốc",
    )
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Chỉ lưu trữ ở local vector database (FAISS/NumPy), không đồng bộ lên Cloud",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Kiểm tra trạng thái kết nối Cloud và dữ liệu Local hiện có",
    )
    args = parser.parse_args()

    pipeline = IngestionPipeline()

    if args.status:
        print(json.dumps(pipeline.get_status(), indent=2, ensure_ascii=False))
        return

    if args.chunk_only:
        chunks = pipeline.run_chunking()
        print(f"=== Đã hoàn tất chunking: {len(chunks)} chunks ===")
        return

    print(f"=== Bắt đầu Ingestion: Store='{args.store}' (Late: {args.late}, BuildGraph: {args.build_graph}, Cloud: {not args.no_cloud}) ===")
    pipeline.run_ingest(
        store_type=args.store,
        late_chunking=args.late,
        sync_to_cloud=not args.no_cloud,
        rechunk=args.rechunk,
        build_graph=args.build_graph,
    )
    print("=== Hoàn tất Ingestion thành công! ===")


if __name__ == "__main__":
    main()
