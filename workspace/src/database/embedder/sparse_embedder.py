from fastembed import SparseTextEmbedding

class SparseEmbedder:
	def __init__(self, model_name):
		self.model = SparseTextEmbedding(model_name=model_name)

	def embed_batch(self, contents):
		embeddings = self.model.embed(contents)
		emb_list = list(embeddings)

		return [
			{
				'indices': emb.indices.tolist(),
				'values': emb.values.tolist()
			}
			for emb in emb_list
		]
	
	def embed_single(self, content):
		embeddings = self.embed_batch([content])
		return embeddings[0]