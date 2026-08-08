import os
from pathlib import Path
from dotenv import load_dotenv

try:
	import torch
except ImportError:
	torch = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / '.env')
load_dotenv(BASE_DIR.parent / '.env')

class Config:
	# DEVICE
	DEVICE = (
		"cuda" if torch and torch.cuda.is_available()
		else "mps" if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
		else "cpu"
	)

	# DATA DIRECTORY
	ORIGINAL_DATA_DIR = BASE_DIR / 'data'
	CHUNKED_DATA_DIR = BASE_DIR / 'data'

	# DATABASE CONFIG
	QDRANT_URL = os.getenv('QDRANT_URL')
	QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
	COLLECTION_NAME = 'landlaw'

	# RETRIEVAL CONFIG
	DENSE_EMBEDDING_MODEL = 'BAAI/bge-m3' # or keepitreal/vietnamese-sbert
	LONG_DENSE_EMBEDDING_MODEL = 'BAAI/bge-m3'
	RERANKER_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
	RETRIEVAL_CANDIDATE_LIMIT = 20
	RERANK_LIMIT = 5
	RETRIEVE_EVALUATOR = 'ntphuc149/ViLegalQwen2.5-1.5B-Base'

	# GROQ CONFIG
	GROQ_API_KEY = os.getenv('GROQ_API_KEY')
	TEST_LLM = 'llama-3.3-70b-versatile'
	TRUE_LLM = 'llama-3.3-70b-versatile'
	SUPPORT_LLM = 'openai/gpt-oss-120b'
	RAGAS_LLM = 'z-ai/glm-5.2'

	# NVIDIA CONFIG
	NVIDIA_KEY = os.getenv('NVIDIA_KEY')
	NVIDIA_BASE_URL = os.getenv('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')

	# EMBEDDING MODEL (centralized)
	EMBEDDING_MODEL = "BAAI/bge-m3" # or keepitreal/vietnamese-sbert

	# EVAL CONFIG
	EVAL_DATA_DIR = BASE_DIR / 'data' / 'eval'
	EVAL_RESULTS_DIR = BASE_DIR / 'data' / 'eval' / 'results'

settings = Config()
