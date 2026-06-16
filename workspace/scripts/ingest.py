from src.configs import settings
from src.data_pipeline.chunker import MDChunker
from src.data_pipeline.embedder import Embedder

import json

def inject(chunk):
	metadata = chunk["metadata"]
	text = f'[{metadata["title"]}][{metadata["chapter"]}][{metadata["article"]}] - {chunk["content"]}'

	return text

def main():
	chunker = MDChunker()
	embedder = Embedder("keepitreal/vietnamese-sbert")

	with open(settings.ORIGINAL_DATA_DIR / "landlaw.md", "r") as file:
		content = file.read()

	chunks = chunker.chunking(content)
	chunker.save_to_json(chunks, settings.CHUNKED_DATA_DIR / "landlaw_chunks.json")

	# Test with 32 chunks
	embeddings = embedder.embed_batch([inject(chunk) for chunk in chunks[:32]])
	with open(settings.CHUNKED_DATA_DIR / "landlaw_embeddings.json", "w", encoding="utf-8") as f:
		json.dump(
			embeddings,
			f,
			ensure_ascii=False,
			indent=4,
		)

	# TODO: Embed all chunks and save to vector database 

if __name__ == "__main__":
	main()