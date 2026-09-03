import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import settings
from src.database.db_manager import DBManager
from src.database.storage import BaseStore, ContrieverStore, Neo4jGraphStore

logger = logging.getLogger(__name__)


class StoreManager:
    """
    Trình quản lý tập trung (Manager) cho toàn bộ Vector Store và Graph Database:
    1. BaseStore: BGE-M3 Dense + BM25 Sparse (Standard & Late Chunking)
       - Local: db/vector_database/baseline/
       - Cloud: Qdrant Cloud (settings.COLLECTION_NAME)
    2. ContrieverStore: mContriever Dense
       - Local: db/vector_database/contriever/
       - Cloud: Qdrant Cloud (settings.CONTRIEVER_COLLECTION_NAME)
    3. Neo4jGraphStore: Knowledge Graph từ GraphRAG Parquet & LanceDB
       - Local: db/graph_database/ (settings.yaml & lancedb/)
       - Cloud: Neo4j Aura (settings.NEO4J_URI)

    Quy tắc cốt lõi: Ưu tiên Local index & embedding > Cloud (settings.PREFER_LOCAL_STORAGE).
    """

    def __init__(
        self,
        prefer_local: Optional[bool] = None,
        db_manager: Optional[DBManager] = None,
        collection_name: Optional[str] = None,
    ):
        self.prefer_local = prefer_local if prefer_local is not None else getattr(settings, "PREFER_LOCAL_STORAGE", True)
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self._db_manager = db_manager

        # Lazy-initialized stores
        self._base_store: Optional[BaseStore] = None
        self._contriever_store: Optional[ContrieverStore] = None
        self._graph_store: Optional[Neo4jGraphStore] = None

    @property
    def db_manager(self) -> DBManager:
        if self._db_manager is None:
            self._db_manager = DBManager(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        return self._db_manager

    @property
    def base(self) -> BaseStore:
        """Truy cập BaseVectorStore (BGE-M3 + BM25)."""
        if self._base_store is None:
            self._base_store = BaseStore(
                db_manager=self.db_manager,
                local_dir=settings.BASELINE_VECTOR_DIR,
                collection_name=self.collection_name,
                prefer_local=self.prefer_local,
            )
        return self._base_store

    @property
    def contriever(self) -> ContrieverStore:
        """Truy cập ContrieverVectorStore (mContriever)."""
        if self._contriever_store is None:
            self._contriever_store = ContrieverStore(
                db_manager=self.db_manager,
                local_dir=settings.CONTRIEVER_VECTOR_DIR,
                collection_name=getattr(settings, "CONTRIEVER_COLLECTION_NAME", "landlaw_contriever"),
                prefer_local=self.prefer_local,
            )
        return self._contriever_store

    @property
    def graph(self) -> Neo4jGraphStore:
        """Truy cập Neo4jGraphStore (GraphRAG -> Neo4j)."""
        if self._graph_store is None:
            self._graph_store = Neo4jGraphStore(
                graph_dir=settings.GRAPH_DB_DIR,
            )
        return self._graph_store

    def get_store(self, store_type: str = "base") -> Union[BaseStore, ContrieverStore, Neo4jGraphStore]:
        """Lấy store theo định danh: 'base', 'contriever', hoặc 'graph'."""
        st = (store_type or "base").lower().strip()
        if st in ("base", "baseline", "default", "qdrant"):
            return self.base
        elif st in ("contriever", "mcontriever"):
            return self.contriever
        elif st in ("graph", "neo4j", "graphrag"):
            return self.graph
        else:
            raise ValueError(f"Không hỗ trợ store_type '{store_type}'. Khả dụng: ['base', 'contriever', 'graph']")

    # ─────────────────────────────────────────────────────────────
    # Unified Querying Interface (Local > Cloud)
    # ─────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        store_type: str = "base",
        limit: int = 20,
        query_filter: Optional[Any] = None,
        prefer_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Truy vấn tìm kiếm thống nhất qua StoreManager:
        - Tự động sinh query embedding theo model của store tương ứng.
        - Tự động ưu tiên local index & embedding > Cloud.
        """
        use_local = prefer_local if prefer_local is not None else self.prefer_local
        st = (store_type or "base").lower().strip()

        if st in ("base", "baseline"):
            query_vector = self.base.dense_embedder.embed_single(query)
            return self.base.query_dense(
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                prefer_local=use_local,
            )
        elif st in ("contriever", "mcontriever"):
            return self.contriever.query(
                query=query,
                limit=limit,
                query_filter=query_filter,
                prefer_local=use_local,
            )
        elif st in ("graph", "neo4j", "graphrag"):
            return self.graph.query_chunks(query=query, limit=limit)
        else:
            raise ValueError(f"Không hỗ trợ store_type: {store_type}")

    # ─────────────────────────────────────────────────────────────
    # Unified Ingestion Pipeline
    # ─────────────────────────────────────────────────────────────

    def chunk(self, input_file: Optional[Path] = None, output_file: Optional[Path] = None):
        """Thực hiện riêng bước chunking từ file Markdown sang file JSON."""
        src_file = Path(input_file or (settings.ORIGINAL_DATA_DIR / "landlaw.md"))
        out_chunks = Path(output_file or (settings.CHUNKED_DATA_DIR / "landlaw_chunks.json"))
        return self.base.chunk_and_save(src_file, out_chunks)

    def build_graph(self, verbose: bool = True) -> int:
        """Kích hoạt Microsoft GraphRAG CLI để trích xuất tri thức và sinh parquet/lancedb."""
        return self.graph.build_graphrag_index(graph_dir=settings.GRAPH_DB_DIR, verbose=verbose)

    def ingest(
        self,
        store_type: str = "base",
        late_chunking: bool = False,
        sync_to_cloud: bool = True,
        rechunk: bool = False,
        build_graph: bool = False,
        input_file: Optional[Path] = None,
        chunks_file: Optional[Path] = None,
    ):
        """
        Thực hiện pipeline nạp dữ liệu:
        - store_type='base': Chuẩn bị chunks và nạp Base Store (standard hoặc late chunking).
        - store_type='contriever': Nạp Contriever Store.
        - store_type='graph': Đồng bộ Parquet & LanceDB lên Neo4j (hoặc chạy build_graph nếu được yêu cầu).
        - store_type='all': Nạp toàn bộ cả 3 hệ thống lưu trữ!
        """
        st = (store_type or "base").lower().strip()
        src_file = Path(input_file or (settings.ORIGINAL_DATA_DIR / "landlaw.md"))
        out_chunks = Path(chunks_file or (settings.CHUNKED_DATA_DIR / "landlaw_chunks.json"))

        # 1. Đảm bảo file chunked JSON tồn tại (hoặc ép chunk lại nếu rechunk=True)
        if st in ("base", "contriever", "all"):
            if not out_chunks.exists() or rechunk:
                logger.info(f"Tiến hành chunking từ {src_file} -> {out_chunks}...")
                self.base.chunk_and_save(src_file, out_chunks)

        # 2. Xây dựng Knowledge Graph qua GraphRAG index nếu được yêu cầu
        if (st in ("graph", "all")) and build_graph:
            logger.info("Kích hoạt GraphRAG index để xây dựng Knowledge Graph...")
            self.build_graph()

        if st in ("base", "all"):
            if late_chunking:
                logger.info("Nạp Base Store bằng LATE CHUNKING...")
                self.base.ingest_documents_late(
                    collection_name=self.collection_name,
                    chunks_json_path=out_chunks,
                    sync_to_cloud=sync_to_cloud,
                )
            else:
                logger.info("Nạp Base Store bằng STANDARD CHUNKING...")
                self.base.ingest_documents(
                    collection_name=self.collection_name,
                    chunks_json_path=out_chunks,
                    sync_to_cloud=sync_to_cloud,
                )

        if st in ("contriever", "all"):
            logger.info("Nạp Contriever Store...")
            self.contriever.ingest_documents(
                chunks_json_path=out_chunks,
                sync_to_cloud=sync_to_cloud,
            )

        if st in ("graph", "all"):
            if not sync_to_cloud:
                logger.info("[Graph] Bỏ qua đồng bộ Neo4j Cloud vì flag --no-cloud được kích hoạt (dữ liệu Graph đã lưu trữ local tại db/graph_database/).")
            else:
                logger.info("Đồng bộ Knowledge Graph lên Neo4j...")
                self.graph.import_all_from_dir()

    # ─────────────────────────────────────────────────────────────
    # Status Diagnostics
    # ─────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Báo cáo trạng thái dữ liệu hiện có ở Local và Cloud cho các store."""
        base_local = self.base.is_local_available()
        contriever_local = self.contriever.is_local_available()

        graph_files = list(settings.GRAPH_DB_DIR.glob("*.parquet")) if settings.GRAPH_DB_DIR.exists() else []
        neo4j_connected = self.graph.driver is not None

        return {
            "priority": "local > cloud" if self.prefer_local else "cloud",
            "base_store": {
                "local_available": base_local,
                "local_dir": str(settings.BASELINE_VECTOR_DIR),
                "collection_name": self.collection_name,
            },
            "contriever_store": {
                "local_available": contriever_local,
                "local_dir": str(settings.CONTRIEVER_VECTOR_DIR),
                "collection_name": getattr(settings, "CONTRIEVER_COLLECTION_NAME", "landlaw_contriever"),
            },
            "graph_store": {
                "local_dir": str(settings.GRAPH_DB_DIR),
                "parquet_files_found": len(graph_files),
                "neo4j_connected": neo4j_connected,
            },
        }
