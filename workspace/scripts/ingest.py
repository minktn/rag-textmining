from xmlrpc import client

from src.configs import settings
from src.data_pipeline.chunker import MDChunker
from src.data_pipeline.embedder import Embedder
from src.database import DBManager

from qdrant_client.models import PointStruct

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
	chunker = MDChunker(chunk_size=1000, chunk_overlap=100)

	with open(settings.ORIGINAL_DATA_DIR / "landlaw.md", "r") as file:
		content = file.read()

	chunks = chunker.chunking(content)
	chunker.save_to_json(chunks, settings.CHUNKED_DATA_DIR / "landlaw_chunks.json")
	print(f"Saved {len(chunks)} chunks to {settings.CHUNKED_DATA_DIR / 'landlaw_chunks.json'}")

def embedding():
	embedder = Embedder("keepitreal/vietnamese-sbert")

	with open(settings.CHUNKED_DATA_DIR / "landlaw_chunks.json", "r") as file:
		chunks = json.load(file)

	embeddings = embedder.embed_batch([enrich_content(chunk) for chunk in chunks])
	
	db_manager = DBManager(
		url=settings.QDRANT_URL,
		api_key=settings.QDRANT_API_KEY
	)

	collection_name = 'landlaw'

	db_manager.setup_collection(collection_name)

	points = []
	for (chunk, embedding) in (zip(chunks, embeddings)):
		payload = chunk.get('metadata', {}).copy()
		payload['content'] = chunk.get('content', '')
		points.append(
			PointStruct(
				id=chunk['id'],
				vector={'dense': embedding},
				payload=payload
			)
		)

	db_manager.upsert_points(collection_name, points)

if __name__ == "__main__":
	# chunking()
	embedding()