from src.data_pipeline.embedder import Embedder
from src.configs import settings
from src.database import DBManager

def main():
	db_manager = DBManager(
		url=settings.QDRANT_URL,
		api_key=settings.QDRANT_API_KEY
	)

	sample_texts = 'Luật đất đai quy định những gì?'

	embedder = Embedder("keepitreal/vietnamese-sbert")
	query_vector = embedder.embed_single(sample_texts)

	search_results = db_manager.query_dense(
		collection_name='landlaw',
		query_vector=query_vector,
		limit=5
	)

	for result in search_results.points:
		print(f"Article: {result.payload.get('article', 'N/A')}")
		print(f"Score: {result.score}")
		print(f"Content: {result.payload['content'][:150]}...")
		print("---")
	
if __name__ == "__main__":
	main()