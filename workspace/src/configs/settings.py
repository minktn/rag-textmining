import os
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Config:
	# DEVICE
	DEVICE = (
		"cuda" if torch.cuda.is_available()
		else "mps" if torch.backends.mps.is_available()
		else "cpu"
	)

	# DATA DIRECTORY
	ORIGINAL_DATA_DIR = BASE_DIR / "data" / "original"
	CHUNKED_DATA_DIR = BASE_DIR / "data" / "chunked"

	# DATABASE CONFIG
	QDRANT_URL = os.getenv("QDRANT_URL")
	QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

settings = Config()