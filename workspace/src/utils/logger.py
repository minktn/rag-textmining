import logging
import sys
from datetime import datetime
from pathlib import Path


class RAGLogger:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"

    def __init__(self, log_dir: str = "logs", name: str = "rag"):
        # Detect if console supports unicode
        self.use_unicode = sys.stdout.encoding.lower() == 'utf-8'
        
        self.SYMB_STEP = "▶" if self.use_unicode else ">"
        self.SYMB_OK = "✔" if self.use_unicode else "OK"
        self.SYMB_WARN = "⚠" if self.use_unicode else "!!"
        self.SYMB_ERR = "✘" if self.use_unicode else "X"
        self.SYMB_LINE = "═" if self.use_unicode else "="
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"rag_{timestamp}.log"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

        self.log_file = log_file
        self._print_raw(f"\n{self.DIM}Log file: {log_file}{self.RESET}\n")

    def _print_raw(self, msg: str):
        print(msg, flush=True)

    def header(self, title: str):
        line = self.SYMB_LINE * 60
        msg = f"\n{self.BOLD}{self.CYAN}{line}\n  {title}\n{line}{self.RESET}"
        self._print_raw(msg)
        self.logger.info(f"{'='*60} {title} {'='*60}")

    def step(self, msg: str):
        self._print_raw(f"{self.BOLD}{self.MAGENTA}{self.SYMB_STEP} {msg}{self.RESET}")
        self.logger.info(f"[STEP] {msg}")

    def info(self, msg: str):
        self._print_raw(f"{self.DIM}  {msg}{self.RESET}")
        self.logger.info(msg)

    def success(self, msg: str):
        self._print_raw(f"{self.GREEN}  {self.SYMB_OK} {msg}{self.RESET}")
        self.logger.info(f"[OK] {msg}")

    def warning(self, msg: str):
        self._print_raw(f"{self.YELLOW}  {self.SYMB_WARN} {msg}{self.RESET}")
        self.logger.warning(msg)

    def error(self, msg: str):
        self._print_raw(f"{self.RED}  {self.SYMB_ERR} {msg}{self.RESET}")
        self.logger.error(msg)

    def result(self, label: str, value: str):
        self._print_raw(f"  {self.WHITE}{label}:{self.RESET} {value}")
        self.logger.info(f"{label}: {value}")
