from src.configs import settings
from src.data_pipeline.chunker import MDChunker

chunker = MDChunker()

with open(settings.ORIGINAL_DATA_DIR / "landlaw.md", "r") as file:
	content = file.read()

chunks = chunker.chunking(content)
chunker.save_to_json(chunks, settings.CHUNKED_DATA_DIR / "landlaw_chunks.json")