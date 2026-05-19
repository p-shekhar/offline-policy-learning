from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter


@dataclass
class ProgressLogger:
    """Small notebook-friendly progress logger."""

    enabled: bool = True
    started_at: float = field(default_factory=perf_counter)

    def _emit(self, prefix: str, message: str) -> None:
        if self.enabled:
            stamp = datetime.utcnow().strftime("%H:%M:%S")
            print(f"[{stamp}] {prefix} {message}", flush=True)

    def step(self, message: str) -> None:
        self._emit(">>", message)

    def log(self, message: str) -> None:
        self._emit("  ", message)

    def done(self, message: str) -> None:
        elapsed = perf_counter() - self.started_at
        self._emit("OK", f"{message} ({elapsed:.1f}s elapsed)")
