import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import settings
from src.common import LegalMetadataProcessor
from src.database.chunker import MDChunker
from src.database.embedder import DenseEmbedder, SparseEmbedder
from src.database import DBManager

from qdrant_client.models import PointStruct, models

import json

def enrich_content(chunk):
	metadata = chunk.get('metadata', {})
	content = chunk.get('content', '')

	if not metadata:
		return content

	metadata_str = ''
	for key, value in metadata.items():
		metadata_str += f"[{key}: {value}] "
	
	enriched_content = f"{metadata_str.strip()}\n{content}"

	return enriched_content

def chunking():
	chunker = MDChunker(chunk_size=1000)

	with open(settings.ORIGINAL_DATA_DIR / "landlaw.md", "r", encoding="utf-8") as file:
		content = file.read()

	chunks = chunker.chunking(content)
	chunker.save_to_json(chunks, settings.CHUNKED_DATA_DIR / "landlaw_chunks.json")
	print(f"Saved {len(chunks)} chunks to {settings.CHUNKED_DATA_DIR / 'landlaw_chunks.json'}")

def embedding():
	dense_embedder = DenseEmbedder("keepitreal/vietnamese-sbert")
	sparse_embedder = SparseEmbedder("Qdrant/bm25")
	metadata_processor = LegalMetadataProcessor()

	with open(settings.CHUNKED_DATA_DIR / "landlaw_chunks.json", "r", encoding="utf-8") as file:
		chunks = json.load(file)

	enriched_chunks = [enrich_content(chunk) for chunk in chunks]

	dense_embeddings = dense_embedder.embed_batch(enriched_chunks)
	sparse_embeddings = sparse_embedder.embed_batch(enriched_chunks)

	db_manager = DBManager(
		url=settings.QDRANT_URL,
		api_key=settings.QDRANT_API_KEY
	)

	collection_name = 'landlaw'

	db_manager.setup_collection(collection_name)

	points = []
	for (chunk, dense_embedding, sparse_embedding) in (zip(chunks, dense_embeddings, sparse_embeddings)):
		content = chunk.get('content', '')
		payload = metadata_processor.enrich_payload(chunk.get('metadata', {}), content)
		payload['content'] = content
		points.append(
			PointStruct(
				id=chunk['id'],
				vector={
					'dense': dense_embedding,
					'sparse': models.SparseVector(
						indices=sparse_embedding['indices'],
						values=sparse_embedding['values']
					)
				},
				payload=payload
			)
		)

	db_manager.upsert_points(collection_name, points)

if __name__ == "__main__":
	# chunking()
	embedding()
