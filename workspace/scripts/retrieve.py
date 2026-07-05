from src.data_pipeline import Embedder
from src.configs import settings
from src.database import DBManager
from src.llm import LLMManager

def main():
	db_manager = DBManager(
		url=settings.QDRANT_URL,
		api_key=settings.QDRANT_API_KEY
	)

	query = 'Luật đất đai quy định những gì?'

	embedder = Embedder("keepitreal/vietnamese-sbert")
	query_vector = embedder.embed_single(query)

	search_results = db_manager.query_dense(
		collection_name='landlaw',
		query_vector=query_vector,
		limit=5
	)

	llm_manager = LLMManager(api_key=settings.GROQ_API_KEY, temperature=0.1)
	prompt = llm_manager.construct_prompt(query, docs=search_results)

	response = llm_manager.generate_response(prompt, model_name=settings.TEST_LLM)
	if response:
		print("Generated Response:")
		print(response)

	
if __name__ == "__main__":
	main()