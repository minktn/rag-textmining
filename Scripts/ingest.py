import argparse
import json
import sys
from pathlib import Path

# Ensure the workspace root is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.database.store_manager import StoreManager


def main():
    parser = argparse.ArgumentParser(description="CLI Data Ingestion & Chunking Pipeline")
    parser.add_argument(
        "--store",
        type=str,
        default="base",
        choices=["base", "contriever", "graph", "all"],
        help="Storage target ('base', 'contriever', 'graph', 'all')",
    )
    parser.add_argument(
        "--late",
        action="store_true",
        help="Use Late Chunking for base store ingestion",
    )
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="Only perform chunking from Markdown to JSON without embedding/ingesting",
    )
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="Force re-chunking even if chunks.json already exists",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Trigger 'graphrag index --root db/graph_database' to build/extract knowledge graph from raw input",
    )
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Do not sync to cloud, save locally only",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current storage status (local & cloud)",
    )
    args = parser.parse_args()

    manager = StoreManager()

    if args.status:
        print(json.dumps(manager.status(), indent=2, ensure_ascii=False))
        return

    if args.chunk_only:
        print("=== Bắt đầu Chunking văn bản luật ===")
        chunks = manager.chunk()
        print(f"=== Đã hoàn tất chunking: {len(chunks)} chunks đã được tạo ===")
        return

    print(f"=== Bắt đầu Ingestion cho Store: '{args.store}' (BuildGraph: {args.build_graph}) ===")
    manager.ingest(
        store_type=args.store,
        late_chunking=args.late,
        sync_to_cloud=not args.no_cloud,
        rechunk=args.rechunk,
        build_graph=args.build_graph,
    )
    print("=== Hoàn tất Ingestion thành công! ===")


if __name__ == "__main__":
    main()
