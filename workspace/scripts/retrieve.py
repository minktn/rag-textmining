import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import settings


def parse_args():
	parser = argparse.ArgumentParser(
		description='Truy xuất ngữ cảnh pháp lý cho câu hỏi.'
	)
	parser.add_argument(
		'--query',
		required=True,
		help='Câu hỏi của người dùng cần truy xuất ngữ cảnh.',
	)
	parser.add_argument(
		'--collection',
		default=settings.COLLECTION_NAME,
		help='Tên collection Qdrant.',
	)
	parser.add_argument(
		'--no-generate',
		action='store_true',
		help='Chỉ in kết quả truy xuất; không gọi LLM.',
	)
	return parser.parse_args()


def build_retriever(collection_name):
	from src.database import DBManager
	from src.retriever import Retriever

	db_manager = DBManager(
		url=settings.QDRANT_URL,
		api_key=settings.QDRANT_API_KEY,
	)

	return Retriever(
		db_manager=db_manager,
		collection_name=collection_name,
		dense_model_name=settings.DENSE_EMBEDDING_MODEL,
		reranker_model_name=settings.RERANKER_MODEL,
		dense_candidate_limit=settings.RETRIEVAL_CANDIDATE_LIMIT,
		rerank_limit=settings.RERANK_LIMIT,
	)


def main():
	args = parse_args()
	retriever = build_retriever(args.collection)
	retrieval_result = retriever.retrieve(args.query)

	if args.no_generate:
		print(json.dumps(retrieval_result, ensure_ascii=False, indent=2))
		return

	from src.llm import LLMManager

	llm_manager = LLMManager(api_key=settings.GROQ_API_KEY, temperature=0.1)
	prompt = llm_manager.construct_prompt(
		retrieval_result['query'],
		docs=retrieval_result['context_chunks'],
	)

	response = llm_manager.generate_response(prompt, model_name=settings.TEST_LLM)
	if response:
		print('Câu trả lời được tạo:')
		print(response)


if __name__ == '__main__':
	main()
