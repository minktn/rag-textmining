from src.data_pipeline import SparseEmbedder
from src.configs import settings

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

embedder = SparseEmbedder(model_name='Qdrant/bm25')

with open(settings.CHUNKED_DATA_DIR / "landlaw_chunks.json", "r") as file:
	chunks = json.load(file)

contents = [enrich_content(chunk) for chunk in chunks[:5]]
embeddings = embedder.embed_batch(contents)

embedding = embeddings[0]

print(type(embedding))
print(type(embedding.indices))
print(type(embedding.values))