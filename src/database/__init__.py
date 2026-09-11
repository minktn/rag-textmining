from .bm25 import BM25Index, BM25Okapi
from .db_manager import DBManager
from .store_manager import StoreManager
from .storage import (
    BaseStore,
    ContrieverStore,
    Neo4jGraphStore,
)


def __getattr__(name: str):
    if name == "IngestionPipeline":
        from .ingest import IngestionPipeline
        return IngestionPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BM25Index",
    "BM25Okapi",
    "DBManager",
    "StoreManager",
    "IngestionPipeline",
    "BaseStore",
    "ContrieverStore",
    "Neo4jGraphStore",
]