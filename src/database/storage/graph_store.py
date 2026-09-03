import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import settings

logger = logging.getLogger(__name__)


def sanitize_list(val: Any) -> List[str]:
    """Chuyển numpy array hoặc list thành python list strings an toàn cho Neo4j."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, np.ndarray)):
        return [str(x) for x in val if x is not None]
    return [str(val)]


class Neo4jGraphStore:
    """
    Quản lý đồng bộ và import dữ liệu Knowledge Graph từ GraphRAG Parquet files lên Neo4j.
    Các thực thể (Entities), Mối quan hệ (Relationships), Đoạn văn bản (TextUnits), và Báo cáo cộng đồng (CommunityReports)
    sẽ được import với cấu trúc Node và Edge tối ưu cho truy vấn đồ thị pháp luật.

    Local path mặc định: db/graph_database/ (kèm lancedb/ và settings.yaml).
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        graph_dir: Optional[Union[str, Path]] = None,
        embedding_model: Optional[Any] = None,
    ):
        self.uri = uri or settings.NEO4J_URI or os.getenv("NEO4J_URI")
        self.username = username or settings.NEO4J_USERNAME or os.getenv("NEO4J_USERNAME")
        self.password = password or settings.NEO4J_PASSWORD or os.getenv("NEO4J_PASSWORD")
        self.database = database or settings.NEO4J_DATABASE or os.getenv("NEO4J_DATABASE", "neo4j")
        self.graph_dir = Path(graph_dir or settings.GRAPH_DB_DIR)
        self.embedding_model = embedding_model
        self.driver = None

        if not self.uri or not self.username or not self.password:
            logger.warning(
                "Thiếu thông tin kết nối Neo4j (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD). "
                "Neo4j driver chưa được kích hoạt."
            )
            return

        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Kết nối Neo4j thành công tại: {self.uri} (Database: {self.database})")
        except Exception as e:
            logger.warning(f"Không thể kết nối Neo4j: {e}")
            self.driver = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            from src.database.embedder import DenseEmbedder
            self.embedding_model = DenseEmbedder(settings.EMBEDDING_MODEL)
        return self.embedding_model

    def _get_lancedb_vectors(self, table_name: str, graph_dir: Optional[Path] = None) -> Dict[str, List[float]]:
        """Đọc trước vector embeddings đã được tính sẵn trong LanceDB để tiết kiệm thời gian."""
        target_dir = graph_dir or self.graph_dir
        lancedb_dir = target_dir / "lancedb"
        if not lancedb_dir.exists():
            return {}
        try:
            import lancedb
            db = lancedb.connect(str(lancedb_dir))
            table_names = db.table_names() if hasattr(db, "table_names") else db.list_tables()
            if table_name in table_names:
                tbl = db.open_table(table_name)
                df = tbl.to_pandas()
                if "id" in df.columns and "vector" in df.columns:
                    logger.info(f"Tìm thấy {len(df)} vector có sẵn trong LanceDB table '{table_name}'. Tái sử dụng ngay.")
                    return {str(r["id"]): [float(v) for v in r["vector"]] for _, r in df.iterrows()}
        except Exception as e:
            logger.warning(f"Không thể đọc vector từ LanceDB ({table_name}): {e}")
        return {}

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Đã đóng kết nối Neo4j.")

    def create_constraints_and_indexes(self):
        """Tạo Constraints, Vector Indexes và Fulltext Indexes trên Neo4j."""
        if not self.driver:
            raise ConnectionError("Neo4j driver chưa được kết nối.")

        queries = [
            # 1. Unique Constraints
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT text_unit_id_unique IF NOT EXISTS FOR (t:TextUnit) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (c:CommunityReport) REQUIRE c.id IS UNIQUE",
            # 2. Standard Indexes
            "CREATE INDEX entity_title_idx IF NOT EXISTS FOR (e:Entity) ON (e.title)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX text_unit_doc_idx IF NOT EXISTS FOR (t:TextUnit) ON (t.document_id)",
            # 3. Vector Indexes (1024 chiều - BAAI/bge-m3, Cosine Similarity)
            """
            CREATE VECTOR INDEX text_unit_embeddings IF NOT EXISTS
            FOR (t:TextUnit)
            ON (t.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
              }
            }
            """,
            """
            CREATE VECTOR INDEX community_report_embeddings IF NOT EXISTS
            FOR (c:CommunityReport)
            ON (c.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
              }
            }
            """,
            # 4. Fulltext Index cho Hybrid Search
            """
            CREATE FULLTEXT INDEX text_unit_fulltext IF NOT EXISTS
            FOR (t:TextUnit)
            ON EACH [t.text]
            """,
        ]
        with self.driver.session(database=self.database) as session:
            for q in queries:
                try:
                    session.run(q.strip())
                except Exception as e:
                    logger.debug(f"Index query note: {e}")
        logger.info("Đã thiết lập xong Neo4j Constraints, Vector & Fulltext Indexes.")

    def clear_database(self):
        """Xóa toàn bộ dữ liệu trong Neo4j (Cẩn trọng)."""
        if not self.driver:
            raise ConnectionError("Neo4j driver chưa được kết nối.")
        logger.warning("Đang xóa toàn bộ dữ liệu hiện có trong Neo4j Database...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Đã dọn dẹp sạch Neo4j Database.")

    def import_text_units(self, parquet_path: Optional[Path] = None, batch_size: int = 200):
        """Import TextUnits từ text_units.parquet vào node (:TextUnit) kèm Dense Vector Embeddings."""
        target_path = Path(parquet_path or (self.graph_dir / "text_units.parquet"))
        if not target_path.exists():
            logger.warning(f"Không tìm thấy file {target_path}")
            return

        df = pd.read_parquet(target_path)
        logger.info(f"Bắt đầu nạp {len(df)} TextUnits...")

        lancedb_vectors = self._get_lancedb_vectors("text_unit_text", target_path.parent)

        missing_indices = [idx for idx, row in df.iterrows() if str(row["id"]) not in lancedb_vectors]
        computed_vectors = {}
        if missing_indices:
            logger.info(f"Tái sử dụng {len(df) - len(missing_indices)} vector từ LanceDB. Cần tính mới {len(missing_indices)} items...")
            emb_model = self._get_embedding_model()
            missing_texts = [str(df.loc[idx].get("text", "")) for idx in missing_indices]
            chunk_size = 32
            for i in tqdm(range(0, len(missing_texts), chunk_size), desc="Calculating Missing Embeddings"):
                sub_texts = missing_texts[i : i + chunk_size]
                sub_vecs = emb_model.embed_batch(sub_texts)
                for j, v in enumerate(sub_vecs):
                    orig_idx = missing_indices[i + j]
                    computed_vectors[str(df.loc[orig_idx]["id"])] = [float(val) for val in v]
        else:
            logger.info(f"Toàn bộ {len(df)} TextUnits đã có sẵn vector từ LanceDB.")

        records = []
        for _, row in df.iterrows():
            cid = str(row["id"])
            vec = lancedb_vectors.get(cid) or computed_vectors.get(cid) or [0.0] * 1024
            records.append({
                "id": cid,
                "human_readable_id": int(row.get("human_readable_id", 0)),
                "text": str(row.get("text", "")),
                "n_tokens": int(row.get("n_tokens", 0)),
                "document_id": str(row.get("document_id", "")),
                "embedding": vec,
            })

        query = """
        UNWIND $batch AS row
        MERGE (t:TextUnit {id: row.id})
        SET t.human_readable_id = row.human_readable_id,
            t.text = row.text,
            t.n_tokens = row.n_tokens,
            t.document_id = row.document_id,
            t.embedding = row.embedding
        """

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Importing TextUnits to Neo4j"):
                batch = records[i : i + batch_size]
                session.run(query, batch=batch)
        logger.info(f"Đã import thành công {len(records)} TextUnits vào Neo4j.")

    def import_entities(self, parquet_path: Optional[Path] = None, batch_size: int = 200):
        """Import Entities từ entities.parquet vào node (:Entity)."""
        target_path = Path(parquet_path or (self.graph_dir / "entities.parquet"))
        if not target_path.exists():
            logger.warning(f"Không tìm thấy file {target_path}")
            return

        df = pd.read_parquet(target_path)
        logger.info(f"Bắt đầu import {len(df)} Entities...")

        records = []
        for _, row in df.iterrows():
            records.append({
                "id": str(row["id"]),
                "human_readable_id": int(row.get("human_readable_id", 0)),
                "title": str(row.get("title", "")).strip(),
                "name": str(row.get("title", "")).strip(),
                "type": str(row.get("type", "UNKNOWN")),
                "description": str(row.get("description", "")),
                "frequency": int(row.get("frequency", 1)),
                "degree": int(row.get("degree", 0)),
                "text_unit_ids": sanitize_list(row.get("text_unit_ids")),
            })

        node_query = """
        UNWIND $batch AS row
        MERGE (e:Entity {name: row.name})
        SET e.id = row.id,
            e.title = row.title,
            e.type = row.type,
            e.description = row.description,
            e.frequency = row.frequency,
            e.degree = row.degree
        """

        rel_query = """
        UNWIND $batch AS row
        MATCH (e:Entity {id: row.id})
        UNWIND row.text_unit_ids AS tu_id
        MATCH (t:TextUnit {id: tu_id})
        MERGE (t)-[:MENTIONS]->(e)
        """

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Importing Entities to Neo4j"):
                batch = records[i : i + batch_size]
                session.run(node_query, batch=batch)
                session.run(rel_query, batch=batch)

        logger.info(f"Đã import thành công {len(records)} Entities và liên kết MENTIONS.")

    def import_relationships(self, parquet_path: Optional[Path] = None, batch_size: int = 200):
        """Import Relationships từ relationships.parquet thành quan hệ [:RELATED_TO] giữa các (:Entity)."""
        target_path = Path(parquet_path or (self.graph_dir / "relationships.parquet"))
        if not target_path.exists():
            logger.warning(f"Không tìm thấy file {target_path}")
            return

        df = pd.read_parquet(target_path)
        logger.info(f"Bắt đầu import {len(df)} Relationships...")

        records = []
        for _, row in df.iterrows():
            records.append({
                "id": str(row["id"]),
                "human_readable_id": int(row.get("human_readable_id", 0)),
                "source": str(row.get("source", "")).strip(),
                "target": str(row.get("target", "")).strip(),
                "description": str(row.get("description", "")),
                "weight": float(row.get("weight", 1.0)),
                "combined_degree": int(row.get("combined_degree", 0)),
                "text_unit_ids": sanitize_list(row.get("text_unit_ids")),
            })

        query = """
        UNWIND $batch AS row
        MERGE (src:Entity {name: row.source})
        MERGE (tgt:Entity {name: row.target})
        MERGE (src)-[r:RELATED_TO {id: row.id}]->(tgt)
        SET r.human_readable_id = row.human_readable_id,
            r.description = row.description,
            r.weight = row.weight,
            r.combined_degree = row.combined_degree,
            r.text_unit_ids = row.text_unit_ids
        """

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Importing Relationships to Neo4j"):
                batch = records[i : i + batch_size]
                session.run(query, batch=batch)

        logger.info(f"Đã import thành công {len(records)} Relationships.")

    def import_community_reports(self, parquet_path: Optional[Path] = None, batch_size: int = 200):
        """Import Community Reports từ community_reports.parquet vào node (:CommunityReport) kèm Embeddings."""
        target_path = Path(parquet_path or (self.graph_dir / "community_reports.parquet"))
        if not target_path.exists():
            logger.warning(f"Không tìm thấy file {target_path}")
            return

        df = pd.read_parquet(target_path)
        logger.info(f"Bắt đầu import {len(df)} Community Reports...")

        lancedb_vectors = self._get_lancedb_vectors("community_full_content", target_path.parent)
        missing_indices = [idx for idx, row in df.iterrows() if str(row["id"]) not in lancedb_vectors]
        computed_vectors = {}
        if missing_indices:
            emb_model = self._get_embedding_model()
            missing_texts = [f"{str(df.loc[idx].get('title', ''))}: {str(df.loc[idx].get('summary', ''))}" for idx in missing_indices]
            chunk_size = 32
            for i in tqdm(range(0, len(missing_texts), chunk_size), desc="Calculating Comm Embeddings"):
                sub_texts = missing_texts[i : i + chunk_size]
                sub_vecs = emb_model.embed_batch(sub_texts)
                for j, v in enumerate(sub_vecs):
                    computed_vectors[str(df.loc[missing_indices[i + j]]["id"])] = [float(val) for val in v]

        records = []
        for _, row in df.iterrows():
            cid = str(row["id"])
            vec = lancedb_vectors.get(cid) or computed_vectors.get(cid) or [0.0] * 1024
            records.append({
                "id": cid,
                "human_readable_id": int(row.get("human_readable_id", 0)),
                "community": int(row.get("community", 0)),
                "level": int(row.get("level", 0)),
                "parent": int(row.get("parent", -1)),
                "title": str(row.get("title", "")),
                "summary": str(row.get("summary", "")),
                "full_content": str(row.get("full_content", "")),
                "rank": float(row.get("rank", 0.0)),
                "rating_explanation": str(row.get("rating_explanation", "")),
                "embedding": vec,
            })

        query = """
        UNWIND $batch AS row
        MERGE (c:CommunityReport {id: row.id})
        SET c.human_readable_id = row.human_readable_id,
            c.community = row.community,
            c.level = row.level,
            c.parent = row.parent,
            c.title = row.title,
            c.summary = row.summary,
            c.full_content = row.full_content,
            c.rank = row.rank,
            c.rating_explanation = row.rating_explanation,
            c.embedding = row.embedding
        """

        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(records), batch_size), desc="Importing Community Reports"):
                batch = records[i : i + batch_size]
                session.run(query, batch=batch)

        logger.info(f"Đã import thành công {len(records)} Community Reports.")

    def import_all_from_dir(self, graph_dir: Optional[Union[str, Path]] = None, clear_first: bool = False):
        """Đồng bộ toàn bộ các bảng Parquet trong thư mục graph_database lên Neo4j."""
        target_dir = Path(graph_dir or self.graph_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục graph database tại: {target_dir}")

        logger.info(f"=== BẮT ĐẦU ĐỒNG BỘ DỮ LIỆU GRAPH TỪ '{target_dir}' LÊN NEO4J ===")

        if clear_first:
            self.clear_database()

        self.create_constraints_and_indexes()
        self.import_text_units(target_dir / "text_units.parquet")
        self.import_entities(target_dir / "entities.parquet")
        self.import_relationships(target_dir / "relationships.parquet")
        self.import_community_reports(target_dir / "community_reports.parquet")
        self.print_graph_stats()
        logger.info("=== HOÀN TẤT ĐỒNG BỘ DỮ LIỆU LÊN NEO4J ===")

    def query_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Thực thi câu truy vấn Cypher và trả về kết quả dưới dạng danh sách dict."""
        if not self.driver:
            raise ConnectionError("Neo4j driver chưa được kết nối.")
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def query_chunks(
        self,
        query: str,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Truy xuất danh sách text chunks liên quan từ Graph Database (LanceDB & text_units.parquet hoặc Neo4j)."""
        chunks: List[Dict[str, Any]] = []

        # 1. Tìm qua LanceDB cục bộ trước (nhanh & độc lập)
        lancedb_dir = self.graph_dir / "lancedb"
        parquet_file = self.graph_dir / "text_units.parquet"

        if lancedb_dir.exists() and parquet_file.exists():
            try:
                import lancedb
                db = lancedb.connect(str(lancedb_dir))
                table_names = db.table_names() if hasattr(db, "table_names") else db.list_tables()
                if "text_unit_text" in table_names:
                    if query_vector is None:
                        embedder = self._get_embedding_model()
                        query_vector = embedder.embed_single(query)

                    tbl = db.open_table("text_unit_text")
                    lancedb_res = tbl.search(query_vector).limit(limit).to_pandas()

                    if not hasattr(self, "_cached_text_units_df") or self._cached_text_units_df is None:
                        self._cached_text_units_df = pd.read_parquet(parquet_file).set_index("id")

                    df_units = self._cached_text_units_df
                    for _, row in lancedb_res.iterrows():
                        unit_id = str(row["id"])
                        dist = float(row.get("_distance", 0.0))
                        score = max(0.0, 1.0 - dist)

                        if unit_id in df_units.index:
                            u_data = df_units.loc[unit_id]
                            text_content = str(u_data.get("text", ""))
                            meta = {
                                "source": "Graph Database",
                                "text_unit_id": unit_id,
                                "document_id": str(u_data.get("document_id", "")),
                                "entity_ids": sanitize_list(u_data.get("entity_ids")),
                                "relationship_ids": sanitize_list(u_data.get("relationship_ids")),
                            }
                            chunks.append({
                                "id": f"graph_{unit_id}",
                                "content": text_content,
                                "dense_score": round(score, 4),
                                "source": "graph_database",
                                "metadata": meta,
                            })

                    if chunks:
                        return chunks
            except Exception as e:
                logger.warning(f"[GraphStore] Lỗi truy vấn qua LanceDB: {e}. Thử fallback...")

        # 2. Fallback qua Neo4j nếu đã kết nối
        if self.driver:
            try:
                if query_vector is None:
                    embedder = self._get_embedding_model()
                    query_vector = embedder.embed_single(query)

                cypher = """
                CALL db.index.vector.queryNodes('text_unit_embeddings', $limit, $vector)
                YIELD node, score
                RETURN node.id AS id, node.text AS text, node.document_id AS document_id, score
                """
                records = self.query_cypher(cypher, {"limit": limit, "vector": query_vector})
                for r in records:
                    chunks.append({
                        "id": f"graph_{r['id']}",
                        "content": r["text"],
                        "dense_score": round(float(r["score"]), 4),
                        "source": "graph_database",
                        "metadata": {
                            "source": "Graph Database (Neo4j)",
                            "text_unit_id": r["id"],
                            "document_id": r.get("document_id", ""),
                        },
                    })
                if chunks:
                    return chunks
            except Exception as e:
                logger.warning(f"[GraphStore] Lỗi truy vấn qua Neo4j: {e}")

        # 3. Fallback đọc trực tiếp text_units.parquet nếu không có vector search
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                for _, row in df.head(limit).iterrows():
                    chunks.append({
                        "id": f"graph_{row['id']}",
                        "content": str(row["text"]),
                        "dense_score": 0.5,
                        "source": "graph_database",
                        "metadata": {"source": "Graph Database (Parquet fallback)"},
                    })
            except Exception as e:
                logger.error(f"[GraphStore] Fallback parquet thất bại: {e}")

        return chunks

    def print_graph_stats(self):
        """In tổng số lượng Node và Relationship hiện có trên Neo4j."""
        if not self.driver:
            return
        with self.driver.session(database=self.database) as session:
            node_counts = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count")
            logger.info("--- Thống kê Node trong Neo4j ---")
            for r in node_counts:
                logger.info(f"Node [:{r['label']}]: {r['count']}")

            rel_counts = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")
            logger.info("--- Thống kê Relationship trong Neo4j ---")
            for r in rel_counts:
                logger.info(f"Relationship [:{r['type']}]: {r['count']}")

    @staticmethod
    def build_graphrag_index(graph_dir: Optional[Union[str, Path]] = None, verbose: bool = True) -> int:
        """Kích hoạt lệnh CLI 'graphrag index --root ...' để trích xuất tri thức và sinh parquet/lancedb."""
        target_dir = str(graph_dir or settings.GRAPH_DB_DIR)
        cmd = [sys.executable, "-m", "graphrag", "index", "--root", target_dir]
        if verbose:
            cmd.append("--verbose")

        logger.info(f"[GraphRAG] Đang chạy lệnh khởi tạo đồ thị: {' '.join(cmd)}")
        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            logger.error(f"[GraphRAG] Lệnh 'graphrag index' kết thúc với mã lỗi: {result.returncode}")
        else:
            logger.info("[GraphRAG] Xây dựng Knowledge Graph hoàn tất thành công.")
        return result.returncode


def sync_graph_to_neo4j(graph_dir: Optional[str] = None, clear_first: bool = False):
    """Hàm helper tiện lợi để gọi từ bên ngoài hoặc CLI."""
    store = Neo4jGraphStore()
    try:
        store.import_all_from_dir(graph_dir=graph_dir, clear_first=clear_first)
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đồng bộ Knowledge Graph từ Parquet (GraphRAG) lên Neo4j")
    parser.add_argument(
        "--graph-dir",
        type=str,
        default=str(settings.GRAPH_DB_DIR),
        help="Đường dẫn đến thư mục graph_database chứa các file parquet",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Xóa sạch dữ liệu cũ trong Neo4j trước khi import",
    )
    args = parser.parse_args()

    sync_graph_to_neo4j(graph_dir=args.graph_dir, clear_first=args.clear)
