import os
import torch
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / '.env')
load_dotenv(BASE_DIR.parent / '.env')

class Config:
	# DEVICE
	DEVICE = (
		"cuda" if torch.cuda.is_available()
		else "mps" if torch.backends.mps.is_available()
		else "cpu"
	)

	# DATA DIRECTORY
	ORIGINAL_DATA_DIR = BASE_DIR / 'data' / 'original'
	CHUNKED_DATA_DIR = BASE_DIR / 'data' / 'chunked'

	# DATABASE CONFIG
	QDRANT_URL = os.getenv('QDRANT_URL')
	QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

	# GROQ CONFIG
	GROQ_API_KEY = os.getenv('GROQ_API_KEY')
	TEST_LLM = 'meta-llama/llama-4-scout-17b-16e-instruct'
	TRUE_LLM = 'llama-3.3-70b-versatile'

	# EMBEDDING MODEL (centralized)
	EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

	# EVAL CONFIG
	EVAL_DATA_DIR = BASE_DIR / 'data' / 'eval'
	EVAL_RESULTS_DIR = BASE_DIR / 'data' / 'eval' / 'results'

settings = Config()