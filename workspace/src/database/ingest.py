import sys
import argparse
from pathlib import Path

# Ensure the workspace root is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.configs import settings
from src.common import LegalMetadataProcessor
from src.database.chunker import MDChunker
from src.database.embedder import DenseEmbedder, SparseEmbedder
from src.database.db_manager import DBManager
from src.database.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Data Ingestion Pipeline")
    parser.add_argument('--late', action='store_true', help="Use Late Chunking for ingestion")
    args = parser.parse_args()

    print("Initializing embedding models and database components...")
    
    # Initialize components
    dense_embedder = DenseEmbedder(settings.DENSE_EMBEDDING_MODEL)
    sparse_embedder = SparseEmbedder("Qdrant/bm25")
    metadata_processor = LegalMetadataProcessor()
    chunker = MDChunker(chunk_size=1000)

    db_manager = DBManager(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )

    # Instantiate the new VectorStore
    vector_store = VectorStore(
        db_manager=db_manager,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        metadata_processor=metadata_processor,
        chunker=chunker
    )

    # Input and output paths
    input_file = settings.ORIGINAL_DATA_DIR / "landlaw.md"
    output_file = settings.CHUNKED_DATA_DIR / "landlaw_chunks.json"
    
    print(f"Checking for existing chunk data at: {output_file}")
    if not output_file.exists():
        print(f"Chunks not found. Chunking {input_file}...")
        vector_store.chunk_and_save(input_file, output_file)

    collection_name = 'landlaw'
    
    if args.late:
        print(f"Starting LATE CHUNKING ingestion into collection '{collection_name}'...")
        vector_store.ingest_documents_late(collection_name, output_file)
    else:
        print(f"Starting standard ingestion into collection '{collection_name}'...")
        vector_store.ingest_documents(collection_name, output_file)

if __name__ == "__main__":
    main()
