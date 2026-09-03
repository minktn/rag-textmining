import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm

from src.config import settings
from src.common import LegalMetadataProcessor
from src.database.chunker import MDChunker
from src.database.embedder import DenseEmbedder, SparseEmbedder
from src.database.db_manager import DBManager

logger = logging.getLogger(__name__)


class VectorStoreBase:
    """Base class cho Vector Storage: lưu trữ Local (NumPy / FAISS / JSON) + fallback Cloud Qdrant."""

    def __init__(
        self,
        local_dir: Path,
        collection_name: str,
        prefer_local: bool = True,
        db_manager: Optional[DBManager] = None,
        metadata_processor: Optional[LegalMetadataProcessor] = None,
    ):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.prefer_local = prefer_local
        self.db_manager = db_manager
        self.metadata_processor = metadata_processor or LegalMetadataProcessor()

        self._local_embeddings: Optional[np.ndarray] = None
        self._local_chunks: Optional[List[Dict[str, Any]]] = None
        self._faiss_index: Optional[Any] = None

    def _get_db_manager(self) -> DBManager:
        if self.db_manager is None:
            self.db_manager = DBManager(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        return self.db_manager

    def _enrich_content(self, chunk: Dict[str, Any]) -> str:
        metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")
        if not metadata:
            return content
        meta_str = " ".join(f"[{k}: {v}]" for k, v in metadata.items() if v not in (None, "", []))
        return f"{meta_str}\n{content}".strip()

    def is_local_available(self) -> bool:
        return (self.local_dir / "dense_embeddings.npy").exists() and (self.local_dir / "chunks.json").exists()

    def save_local(
        self,
        chunks: List[Dict[str, Any]],
        dense_embeddings: Union[List[List[float]], np.ndarray],
        sparse_embeddings: Optional[List[Dict[str, Any]]] = None,
    ):
        self.local_dir.mkdir(parents=True, exist_ok=True)
        emb_array = np.array(dense_embeddings, dtype=np.float32)
        np.save(self.local_dir / "dense_embeddings.npy", emb_array)

        with open(self.local_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        if sparse_embeddings:
            with open(self.local_dir / "sparse_embeddings.json", "w", encoding="utf-8") as f:
                json.dump(sparse_embeddings, f, ensure_ascii=False)

        try:
            import faiss
            dim = emb_array.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(emb_array)
            index.add(emb_array)
            faiss.write_index(index, str(self.local_dir / "index.faiss"))
            self._faiss_index = index
        except Exception:
            pass

        self._local_embeddings = emb_array
        self._local_chunks = chunks
        logger.info(f"Saved {len(chunks)} chunks & embeddings to {self.local_dir}")

    def load_local(self) -> bool:
        if not self.is_local_available():
            return False
        if self._local_embeddings is None:
            self._local_embeddings = np.load(self.local_dir / "dense_embeddings.npy")
        if self._local_chunks is None:
            with open(self.local_dir / "chunks.json", "r", encoding="utf-8") as f:
                self._local_chunks = json.load(f)
        faiss_file = self.local_dir / "index.faiss"
        if faiss_file.exists() and self._faiss_index is None:
            try:
                import faiss
                self._faiss_index = faiss.read_index(str(faiss_file))
            except Exception:
                self._faiss_index = None
        return True

    def query_local(
        self,
        query_vector: List[float],
        limit: int = 20,
        query_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.load_local():
            raise FileNotFoundError(f"No local index at {self.local_dir}")

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        emb_matrix = self._local_embeddings
        emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_norms[emb_norms == 0] = 1.0
        scores = np.dot(emb_matrix / emb_norms, q_vec)

        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            chunk = self._local_chunks[idx]
            metadata = chunk.get("metadata", {})
            if query_filter and any(metadata.get(k) != v for k, v in query_filter.items()):
                continue
            results.append({
                "id": str(chunk.get("id", idx)),
                "score": float(scores[idx]),
                "content": chunk.get("content", ""),
                "metadata": metadata,
                "payload": metadata,
            })
            if len(results) >= limit:
                break
        return results

    def query_dense(
        self,
        query_vector: List[float],
        limit: int = 20,
        query_filter: Optional[Any] = None,
        prefer_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        use_local = prefer_local if prefer_local is not None else self.prefer_local
        if use_local and self.is_local_available():
            try:
                dict_filter = query_filter if isinstance(query_filter, dict) else None
                return self.query_local(query_vector, limit=limit, query_filter=dict_filter)
            except Exception as e:
                logger.warning(f"Local query error: {e}. Fallback to Cloud Qdrant.")

        return self._get_db_manager().query_dense(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
        )


class BaseStore(VectorStoreBase):
    """Base Embeddings Store: BGE-M3 Dense + BM25 Sparse, Standard & Late Chunking."""

    def __init__(
        self,
        db_manager: Optional[DBManager] = None,
        dense_embedder: Optional[DenseEmbedder] = None,
        sparse_embedder: Optional[SparseEmbedder] = None,
        metadata_processor: Optional[LegalMetadataProcessor] = None,
        chunker: Optional[Any] = None,
        local_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        prefer_local: bool = True,
    ):
        super().__init__(
            local_dir=Path(local_dir or settings.BASELINE_VECTOR_DIR),
            collection_name=collection_name or settings.COLLECTION_NAME,
            prefer_local=prefer_local,
            db_manager=db_manager,
            metadata_processor=metadata_processor,
        )
        self._dense_embedder = dense_embedder
        self._sparse_embedder = sparse_embedder
        self.chunker = chunker or MDChunker(chunk_size=1000)

    @property
    def dense_embedder(self) -> DenseEmbedder:
        if self._dense_embedder is None:
            self._dense_embedder = DenseEmbedder(settings.EMBEDDING_MODEL)
        return self._dense_embedder

    @property
    def sparse_embedder(self) -> SparseEmbedder:
        if self._sparse_embedder is None:
            self._sparse_embedder = SparseEmbedder("Qdrant/bm25")
        return self._sparse_embedder

    def chunk_and_save(self, input_file: Optional[Path] = None, output_file: Optional[Path] = None):
        src_file = Path(input_file or (settings.ORIGINAL_DATA_DIR / "landlaw.md"))
        out_file = Path(output_file or (settings.CHUNKED_DATA_DIR / "landlaw_chunks.json"))
        with open(src_file, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = self.chunker.chunking(content)
        self.chunker.save_to_json(chunks, out_file)
        logger.info(f"Saved {len(chunks)} chunks to {out_file}")
        return chunks

    def ingest_documents(
        self,
        collection_name: Optional[str] = None,
        chunks_json_path: Optional[Path] = None,
        sync_to_cloud: bool = True,
    ):
        col_name = collection_name or self.collection_name
        json_path = Path(chunks_json_path or (settings.CHUNKED_DATA_DIR / "landlaw_chunks.json"))
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        enriched = [self._enrich_content(c) for c in chunks]
        dense_embs = self.dense_embedder.embed_batch(enriched)
        sparse_embs = self.sparse_embedder.embed_batch(enriched)

        self.save_local(chunks, dense_embs, sparse_embs)

        if sync_to_cloud and settings.QDRANT_URL:
            try:
                from qdrant_client.models import PointStruct, models
                db = self._get_db_manager()
                db.setup_collection(col_name, vector_size=len(dense_embs[0]), recreate=True)
                points = [
                    PointStruct(
                        id=chunk["id"],
                        vector={
                            "dense": dense_embs[i],
                            "sparse": models.SparseVector(
                                indices=sparse_embs[i]["indices"],
                                values=sparse_embs[i]["values"],
                            ),
                        },
                        payload={
                            **self.metadata_processor.enrich_payload(chunk.get("metadata", {}), chunk.get("content", "")),
                            "content": chunk.get("content", ""),
                        },
                    )
                    for i, chunk in enumerate(chunks)
                ]
                db.upsert_points(col_name, points)
            except Exception as e:
                logger.warning(f"Could not sync to Qdrant: {e}")

    def ingest_documents_late(
        self,
        collection_name: Optional[str] = None,
        chunks_json_path: Optional[Path] = None,
        max_tokens: int = 8192,
        sync_to_cloud: bool = True,
    ):
        from transformers import AutoTokenizer, AutoModel
        import torch
        from src.database.chunker.late_chunking import chunked_pooling

        col_name = collection_name or self.collection_name
        json_path = Path(chunks_json_path or (settings.CHUNKED_DATA_DIR / "landlaw_chunks.json"))
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        device = settings.DEVICE
        tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(settings.EMBEDDING_MODEL).to(device)
        model.eval()

        batches, cur_batch, cur_len = [], [], 0
        for chunk in chunks:
            chunk["enriched_content"] = self._enrich_content(chunk)
            t_len = len(tokenizer.tokenize(chunk["enriched_content"])) + 2
            if cur_len + t_len > max_tokens - 100 and cur_batch:
                batches.append(cur_batch)
                cur_batch, cur_len = [chunk], t_len
            else:
                cur_batch.append(chunk)
                cur_len += t_len
        if cur_batch:
            batches.append(cur_batch)

        all_pooled, all_sparse = [], []
        for batch in tqdm(batches, desc="Late Chunking"):
            text = "\n".join(c["enriched_content"] for c in batch) + "\n"
            spans, pos = [], 0
            for c in batch:
                spans.append((pos, pos + len(c["enriched_content"])))
                pos += len(c["enriched_content"]) + 1

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
            with torch.no_grad():
                out = model(**inputs).last_hidden_state

            span_toks = []
            for s, e in spans:
                t_s = inputs.char_to_token(0, s) or 1
                t_e = (inputs.char_to_token(0, max(s, e - 1)) or t_s) + 1
                span_toks.append((t_s, max(t_s + 1, t_e)))

            pooled = chunked_pooling([out], [span_toks], max_length=max_tokens)[0]
            all_pooled.extend(pooled.cpu().numpy())
            all_sparse.extend(self.sparse_embedder.embed_batch([c["enriched_content"] for c in batch]))

        self.save_local(chunks, all_pooled, all_sparse)

        if sync_to_cloud and settings.QDRANT_URL:
            try:
                from qdrant_client.models import PointStruct, models
                db = self._get_db_manager()
                db.setup_collection(col_name, vector_size=model.config.hidden_size, recreate=True)
                points = [
                    PointStruct(
                        id=chunk["id"],
                        vector={
                            "dense": all_pooled[i].tolist(),
                            "sparse": models.SparseVector(
                                indices=all_sparse[i]["indices"],
                                values=all_sparse[i]["values"],
                            ),
                        },
                        payload={
                            **self.metadata_processor.enrich_payload(chunk.get("metadata", {}), chunk.get("content", "")),
                            "content": chunk.get("content", ""),
                        },
                    )
                    for i, chunk in enumerate(chunks)
                ]
                db.upsert_points(col_name, points)
            except Exception as e:
                logger.warning(f"Could not sync to Qdrant: {e}")
