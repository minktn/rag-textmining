import threading
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, Document

class DBManager:
	_instance = None
	_lock = threading.Lock()

	def __new__(cls, *args, **kwargs):
		if not cls._instance:
			with cls._lock:
				if not cls._instance:
					cls._instance = super(DBManager, cls).__new__(cls)
					cls._instance._initialized = False
		
		return cls._instance
	
	def __init__(self, url, api_key):
		if self._initialized:
			return
		
		self.client = QdrantClient(
			url=url,
			api_key=api_key,
			check_compatibility=False
		)

		self._initialized = True

	def setup_collection(self, collection_name, vector_size=768, distance_metric=Distance.COSINE):
		if not self.client.collection_exists(collection_name):
			self.client.create_collection(
				collection_name=collection_name,
				vectors_config={
					'dense': VectorParams(size=vector_size, distance=distance_metric)
				},

				# For future use (hybrid search)
				sparse_vectors_config={
					'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)
				}
			)

	def upsert_points(self, collection_name, points, batch_size=100):
		if points:
			for i in range(0, len(points), batch_size):
				batch = points[i : i + batch_size]
				self.client.upsert(
					collection_name=collection_name,
					points=batch
				)
	
	def query_dense(self, collection_name, query_vector, limit=5):
		search_results = self.client.query_points(
			collection_name=collection_name,
			query=query_vector,
			using='dense',
			limit=limit
		)

		formatted_results = [
			{'content': point.payload.get('content', '')}
			for point in search_results.points
		]
		return formatted_results

	# def query_sparse(self, collection_name, query_filter, limit=5):
	# 	search_results = self.client.search(
	# 		collection_name=collection_name,
	# 		query_filter=query_filter,
	# 		limit=limit
	# 	)

	# 	return search_results