import sys
from typing import Optional


class SectionProgress:
    BAR_WIDTH = 35

    def __init__(self, label: str, logger=None):
        self.label = label
        self.logger = logger
        self._last_pct = -1

    def update(self, current: int, total: int, status: str = ""):
        if total == 0:
            pct = 100
        else:
            pct = int((current / total) * 100)

        filled = int(self.BAR_WIDTH * pct / 100)
        bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)

        line = f"\r  [{bar}] {pct:3d}%  {status[:45]:<45}"
        sys.stdout.write(line)
        sys.stdout.flush()

        if self.logger and pct != self._last_pct:
            # Note: Assuming self.logger is an instance of RAGLogger which has a .logger attribute
            if hasattr(self.logger, "logger"):
                self.logger.logger.debug(f"[PROGRESS] {self.label} {pct}% — {status}")
            self._last_pct = pct

        if pct == 100:
            sys.stdout.write("\n")
            sys.stdout.flush()


class ProgressTracker:
    def __init__(self, logger=None):
        self.logger = logger

    def section(self, label: str) -> SectionProgress:
        if self.logger:
            self.logger.info(f"Progress: {label}")
        return SectionProgress(label=label, logger=self.logger)
