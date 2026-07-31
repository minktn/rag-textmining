import json

class BaseChunker:
	def chunking(self, text):
		raise NotImplementedError("Chunking method must be implemented by subclass.")
	
	def save_to_json(self, chunks, file_path):
		with open(file_path, 'w', encoding="utf-8") as f:
			json.dump(
				chunks,
				f,
				ensure_ascii=False,
				indent=4,
			)

	def load_from_json(self, file_path):
		with open(file_path, 'r', encoding="utf-8") as f:
			chunks = json.load(f)
		return chunks