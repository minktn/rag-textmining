from sentence_transformers import SentenceTransformer

class Embedder:
	def __init__(self, model_name):
		self.model = SentenceTransformer(model_name)

	def embed_batch(self, contents, batch_size=32):
		embeddings = self.model.encode(
			inputs=contents,
			batch_size=batch_size,
			show_progress_bar=False,
			convert_to_numpy=True
		)
		return embeddings.tolist()
	
	def embed_single(self, content):
		embeddings = self.model.encode(content, convert_to_numpy=True)
		return embeddings.tolist()