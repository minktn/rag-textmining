"""
NLI Provider Logging Configuration
==================================
Cấu hình logging cho FastAPI NLI Service (port 8001).
Lưu trữ log xoay vòng tại src/models/nli_model/logs/nli_provider.log
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Thư mục logs cục bộ của nli_model
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "nli_provider.log"

logger = logging.getLogger("nli_provider")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Handler xoay vòng file (Max 10MB, giữ lại 5 files backup)
    file_handler = RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
