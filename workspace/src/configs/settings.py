import os
from pathlib import Path

try:
	import torch
except ImportError:
	torch = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Config:
	# DEVICE
	DEVICE = (
		"cuda" if torch and torch.cuda.is_available()
		else "mps" if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
		else "cpu"
	)

	# DATA DIRECTORY
	ORIGINAL_DATA_DIR = BASE_DIR / 'data' / 'original'
	CHUNKED_DATA_DIR = BASE_DIR / 'data' / 'chunked'

	# DATABASE CONFIG
	QDRANT_URL = os.getenv('QDRANT_URL')
	QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
	COLLECTION_NAME = 'landlaw'

	# RETRIEVAL CONFIG
	DENSE_EMBEDDING_MODEL = 'keepitreal/vietnamese-sbert'
	RERANKER_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
	RETRIEVAL_CANDIDATE_LIMIT = 20
	RERANK_LIMIT = 5

	# GROQ CONFIG
	GROQ_API_KEY = os.getenv('GROQ_API_KEY')
	TEST_LLM = 'meta-llama/llama-4-scout-17b-16e-instruct'
	TRUE_LLM = 'llama-3.3-70b-versatile'

settings = Config()
