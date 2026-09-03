import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.config import settings
from src.common import LegalMetadataProcessor
from src.database.db_manager import DBManager
from .base_store import VectorStoreBase

logger = logging.getLogger(__name__)


class ContrieverStore(VectorStoreBase):
    """Contriever Embeddings Store (facebook/mcontriever-msmarco) kế thừa VectorStoreBase."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        db_manager: Optional[DBManager] = None,
        metadata_processor: Optional[LegalMetadataProcessor] = None,
        local_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        prefer_local: bool = True,
    ):
        super().__init__(
            local_dir=Path(local_dir or settings.CONTRIEVER_VECTOR_DIR),
            collection_name=collection_name or getattr(settings, "CONTRIEVER_COLLECTION_NAME", "landlaw_contriever"),
            prefer_local=prefer_local,
            db_manager=db_manager,
            metadata_processor=metadata_processor,
        )
        self.model_name = model_name or getattr(settings, "CONTRIEVER_MODEL", "facebook/mcontriever-msmarco")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Fallback to HF Transformers for Contriever ({e})")
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._hf_model = AutoModel.from_pretrained(self.model_name).to(settings.DEVICE)
                self._hf_model.eval()
                self._model = "hf"
        return self._model

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.model != "hf":
            return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

        import torch
        device = settings.DEVICE
        all_embs = []
        for i in range(0, len(texts), batch_size):
            inputs = self._tokenizer(texts[i : i + batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self._hf_model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                mean = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                all_embs.append(torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy())
        return np.vstack(all_embs)

    def embed_query(self, query: str) -> List[float]:
        return self.embed_batch([query])[0].tolist()

    def query(
        self,
        query: Union[str, List[float]],
        limit: int = 20,
        query_filter: Optional[Any] = None,
        prefer_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        query_vec = self.embed_query(query) if isinstance(query, str) else query
        return self.query_dense(query_vec, limit=limit, query_filter=query_filter, prefer_local=prefer_local)

    def ingest_documents(
        self,
        chunks_json_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        sync_to_cloud: bool = True,
    ):
        col_name = collection_name or self.collection_name
        json_path = Path(chunks_json_path or (settings.CHUNKED_DATA_DIR / "landlaw_chunks.json"))
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        texts = [self._enrich_content(c) for c in chunks]
        embeddings = self.embed_batch(texts)

        self.save_local(chunks, embeddings)

        if sync_to_cloud and settings.QDRANT_URL:
            try:
                from qdrant_client.models import PointStruct
                db = self._get_db_manager()
                db.setup_collection(col_name, vector_size=embeddings.shape[1], recreate=True)
                points = [
                    PointStruct(
                        id=chunk["id"],
                        vector={"dense": embeddings[i].tolist()},
                        payload={
                            **self.metadata_processor.enrich_payload(chunk.get("metadata", {}), chunk.get("content", "")),
                            "content": chunk.get("content", ""),
                        },
                    )
                    for i, chunk in enumerate(chunks)
                ]
                db.upsert_points(col_name, points)
            except Exception as e:
                logger.warning(f"Could not sync Contriever to Qdrant: {e}")
