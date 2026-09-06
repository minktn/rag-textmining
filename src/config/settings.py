import os
import sys
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
	try:
		sys.stdout.reconfigure(encoding="utf-8", errors="replace")
	except Exception:
		pass
if hasattr(sys.stderr, "reconfigure"):
	try:
		sys.stderr.reconfigure(encoding="utf-8", errors="replace")
	except Exception:
		pass

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
	DB_DIR = BASE_DIR / 'db'
	ORIGINAL_DATA_DIR = DB_DIR / 'input' if (DB_DIR / 'input').exists() else DB_DIR
	CHUNKED_DATA_DIR = DB_DIR

	# LOCAL STORAGE PATHS (FAISS, LanceDB, Local Vector DB)
	LOCAL_VECTOR_DB_DIR = DB_DIR / 'vector_database'
	BASELINE_VECTOR_DIR = LOCAL_VECTOR_DB_DIR / 'baseline'
	CONTRIEVER_VECTOR_DIR = LOCAL_VECTOR_DB_DIR / 'contriever'

	# GRAPH DATABASE (Neo4j & Local LanceDB / Parquet)
	GRAPH_DB_DIR = DB_DIR / 'graph_database'
	GRAPH_SETTINGS_YAML = GRAPH_DB_DIR / 'settings.yaml'
	GRAPH_LANCEDB_DIR = GRAPH_DB_DIR / 'lancedb'

	# DATABASE CONFIG
	QDRANT_URL = os.getenv('QDRANT_URL')
	QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
	COLLECTION_NAME = 'landlaw'
	CONTRIEVER_COLLECTION_NAME = 'landlaw_contriever'

	# NEO4J CONFIG
	NEO4J_URI = os.getenv('NEO4J_URI')
	NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
	NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
	NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')
	AURA_INSTANCEID = os.getenv('AURA_INSTANCEID')
	AURA_INSTANCENAME = os.getenv('AURA_INSTANCENAME')

	# STORAGE PRIORITY: Local index & embedding > Cloud
	PREFER_LOCAL_STORAGE = True

	# RETRIEVAL CONFIG
	RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
	RETRIEVAL_CANDIDATE_LIMIT = 20
	RERANK_LIMIT = 5

	# LLM API CONFIG
	NVIDIA_KEY = os.getenv('NVIDIA_KEY')
	NVIDIA_BASE_URL = os.getenv('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
	GROQ_KEY = os.getenv('GROQ_API_KEY')
	GEMINI_KEY = os.getenv('GOOGLE_API_KEY')

	# LLM MODEL
	GROQ_LLM = 'llama-3.3-70b-versatile'
	NVIDIA_LLM = 'nvidia/nemotron-3-ultra-550b-a55b'
	GEMINI_LLM = 'gemma-4-31b-it'
	LOCAL_LLM = 'ntphuc149/ViLegalQwen2.5-1.5B-Base'
	BASE_TEMP = 0.0
	BASE_MAX_TOKENS = 4096
	REASONING_TEMP = 0.4
	REASONING_MAX_TOKENS = 8192
	LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', 60))


	# EMBEDDING MODEL (centralized)
	EMBEDDING_MODEL = "BAAI/bge-m3"
	CONTRIEVER_MODEL = "facebook/mcontriever-msmarco"

	LLM_SERVICE = "google" # Or "nvidia"/"groq"/"local"
	LLM_MODE = "reason"  # Or "base"

	SUB_LLM_SERVICE = "google" # Or "nvidia"/"groq"/"local"
	SUB_LLM_MODE = "reason"  # Or "base"

	# NLI SERVICE CONFIG
	NLI_API_URL = os.getenv('NLI_API_URL', 'http://localhost:8001')

	# EVAL CONFIG
	EVAL_DATA_DIR = DB_DIR / 'eval'
	EVAL_RESULTS_DIR = DB_DIR / 'results'
	RAGAS_SERVICE = "nvidia" # Or "google"/"groq"
	EVAL_BATCH_SIZE = 10
	EVAL_MAX_WORKERS = 4
	RAGAS_MAX_WORKERS = 4

	ADAPTABLE_POSTPROCESS = ["prompt_compression"]
	ADAPTABLE_PREPROCESS = []

settings = Config()
