import threading
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

class DBManager:
	_instance = None
	_lock = threading.Lock()
	PAYLOAD_INDEX_SCHEMA = {
		'article_no': models.PayloadSchemaType.INTEGER,
		'chapter_no': models.PayloadSchemaType.INTEGER,
		'section_no': models.PayloadSchemaType.INTEGER,
		'clause_nos': models.PayloadSchemaType.INTEGER,
		'ref_article_nos': models.PayloadSchemaType.INTEGER,
	}

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

	def setup_collection(self, collection_name, vector_size=768, distance_metric=Distance.COSINE, recreate=False):
		if recreate and self.client.collection_exists(collection_name):
			self.client.delete_collection(collection_name)

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
		self.ensure_payload_indexes(collection_name)

	def ensure_payload_indexes(self, collection_name):
		collection_info = self.client.get_collection(collection_name)
		payload_schema = getattr(collection_info, 'payload_schema', {}) or {}

		for field_name, field_schema in self.PAYLOAD_INDEX_SCHEMA.items():
			if field_name in payload_schema:
				continue
			self.client.create_payload_index(
				collection_name=collection_name,
				field_name=field_name,
				field_schema=field_schema
			)

	def upsert_points(self, collection_name, points, batch_size=100):
		if points:
			for i in range(0, len(points), batch_size):
				batch = points[i : i + batch_size]
				self.client.upsert(
					collection_name=collection_name,
					points=batch
				)
	
	def query_dense(self, collection_name, query_vector, limit=20, query_filter=None):
		search_results = self.client.query_points(
			collection_name=collection_name,
			query=query_vector,
			using='dense',
			limit=limit,
			query_filter=query_filter
		)

		return [self._format_point(point) for point in search_results.points]

	def fetch_by_filter(self, collection_name, query_filter, limit=20):
		scroll_results, _ = self.client.scroll(
			collection_name=collection_name,
			scroll_filter=query_filter,
			limit=limit,
			with_payload=True,
			with_vectors=False
		)

		return [self._format_point(point) for point in scroll_results]

	def fetch_by_article_no(self, collection_name, article_no, limit=20):
		query_filter = Filter(
			must=[
				FieldCondition(
					key='article_no',
					match=MatchValue(value=article_no)
				)
			]
		)
		return self.fetch_by_filter(collection_name, query_filter, limit=limit)

	def _format_point(self, point):
		payload = point.payload or {}
		return {
			'id': str(point.id),
			'score': getattr(point, 'score', None),
			'content': payload.get('content', ''),
			'metadata': payload,
			'payload': payload,
		}

	# def query_sparse(self, collection_name, query_filter, limit=5):
	# 	search_results = self.client.search(
	# 		collection_name=collection_name,
	# 		query_filter=query_filter,
	# 		limit=limit
	# 	)

	# 	return search_results
