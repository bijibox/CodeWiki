"""
Unified logging utilities for CLI and backend runtime output.
"""

from __future__ import annotations

from datetime import datetime
import logging
import sys
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler

MAX_VERBOSITY = 4
LOGGER_NAME = "codewiki"


def normalize_verbosity(verbosity: int) -> int:
    """Clamp verbosity to the supported range."""
    return max(0, min(verbosity, MAX_VERBOSITY))


class VerbosityFilter(logging.Filter):
    """Filter records according to CLI verbosity and record metadata."""

    def __init__(self, verbosity: int):
        super().__init__()
        self.verbosity = normalize_verbosity(verbosity)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True

        gate = getattr(record, "verbosity_gate", None)
        if isinstance(gate, int):
            return self.verbosity >= gate

        if record.name.startswith(f"{LOGGER_NAME}.src.be"):
            return self.verbosity >= 1

        if record.levelno == logging.DEBUG:
            return self.verbosity >= 1

        return True


class CodeWikiFormatter(logging.Formatter):
    """Formatter for user-facing CLI and detailed LLM output."""

    def format(self, record: logging.LogRecord) -> str:
        event_type = getattr(record, "event_type", "message")
        message = record.getMessage()

        if event_type == "blank":
            return ""
        if event_type == "step":
            step = getattr(record, "step", None)
            total = getattr(record, "total", None)
            prefix = f"[{step}/{total}]" if step is not None and total is not None else "→"
            return f"{prefix} {message}"
        if event_type == "success":
            return f"✓ {message}"
        if event_type == "warning":
            return f"⚠️  {message}"
        if event_type == "error":
            return f"✗ {message}"
        if event_type == "debug":
            return f"[{self._format_clock(record)}] {message}"
        if event_type == "progress_stage":
            if getattr(record, "verbosity_gate", 0) >= 1:
                return (
                    f"\n[{getattr(record, 'elapsed', self._format_clock(record))}] "
                    f"Phase {getattr(record, 'step', '?')}/{getattr(record, 'total', '?')}: {message}"
                )
            return f"[{getattr(record, 'step', '?')}/{getattr(record, 'total', '?')}] {message}"
        if event_type == "progress_update":
            return f"[{getattr(record, 'elapsed', self._format_clock(record))}]   {message}"
        if event_type == "progress_complete":
            suffix = ""
            if getattr(record, "stage_time", None) is not None:
                suffix = f" ({getattr(record, 'stage_time'):.1f}s)"
            return (
                f"[{getattr(record, 'elapsed', self._format_clock(record))}]   "
                f"{message}{suffix}"
            )
        if event_type == "module_progress":
            current = getattr(record, "current", "?")
            total = getattr(record, "total", "?")
            module_type = getattr(record, "module_type", "module")
            module_path = getattr(record, "module_path", message)
            status = getattr(record, "status", "generated")
            suffix = f"... {status}" if status else ""
            return f"  [{current}/{total}] {module_type} {module_path}{suffix}"
        if event_type == "llm_summary":
            phase = getattr(record, "llm_phase", "request")
            if phase == "request":
                request_tokens = getattr(record, "llm_request_tokens", None)
                request_tokens_display = "unknown"
                if request_tokens is not None:
                    request_tokens_display = str(request_tokens)
                return (
                    f"[{self._format_clock(record)}] "
                    f"LLM request: type={getattr(record, 'llm_prompt_type', 'llm')} "
                    f"input_tokens={request_tokens_display}"
                )
            duration_seconds = getattr(record, "llm_duration_seconds", None)
            response_tokens = getattr(record, "llm_response_tokens", None)
            response_tokens_per_second = getattr(record, "llm_response_tokens_per_second", None)
            duration_display = "unknown"
            response_tokens_display = "unknown"
            response_speed_display = "unknown"
            if duration_seconds is not None:
                duration_display = f"{duration_seconds:.3f}"
            if response_tokens is not None:
                response_tokens_display = str(response_tokens)
            if response_tokens_per_second is not None:
                response_speed_display = f"{response_tokens_per_second:.3f}"
            return (
                f"[{self._format_clock(record)}] LLM response: "
                f"duration_s={duration_display} "
                f"output_tokens={response_tokens_display} "
                f"output_tps={response_speed_display}"
            )
        if event_type == "llm_content":
            lines = ["", f"===== {getattr(record, 'llm_title', 'LLM')} ====="]
            prompt_type = getattr(record, "llm_prompt_type", None)
            context = getattr(record, "llm_context", None)
            model = getattr(record, "llm_model", None)
            if prompt_type:
                lines.append(f"type: {prompt_type}")
            if context:
                lines.append(f"context: {context}")
            if model:
                lines.append(f"model: {model}")
            lines.append("----- BEGIN CONTENT -----")
            lines.append(message)
            lines.append("----- END CONTENT -----")
            return "\n".join(lines)
        if event_type == "section":
            return message

        return message

    def _format_clock(self, record: logging.LogRecord) -> str:
        return datetime.fromtimestamp(record.created).strftime("%H:%M:%S")


def configure_logging(verbosity: int = 0) -> logging.Logger:
    """Configure the shared CodeWiki logger hierarchy."""
    normalized = normalize_verbosity(verbosity)
    root_logger = logging.getLogger(LOGGER_NAME)
    root_logger.setLevel(logging.DEBUG)
    root_logger.propagate = False
    root_logger.handlers.clear()

    console = Console(file=sys.stdout, force_terminal=False, color_system="auto", highlight=False)
    handler = RichHandler(
        console=console,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(CodeWikiFormatter())
    handler.addFilter(VerbosityFilter(normalized))
    root_logger.addHandler(handler)

    backend_logger = logging.getLogger(f"{LOGGER_NAME}.src.be")
    backend_logger.handlers.clear()
    backend_logger.setLevel(logging.DEBUG)
    backend_logger.propagate = True

    return root_logger


class CLILogger:
    """Thin wrapper around the shared logger with convenience helpers."""

    def __init__(self, verbosity: int = 0, name: str = f"{LOGGER_NAME}.cli"):
        self.verbosity = normalize_verbosity(verbosity)
        configure_logging(self.verbosity)
        self.logger = logging.getLogger(name)
        self.start_time = datetime.now()

    def info(self, message: str, *, verbosity_gate: int = 0):
        self._log(logging.INFO, message, event_type="message", verbosity_gate=verbosity_gate)

    def debug(self, message: str):
        self._log(logging.DEBUG, message, event_type="debug", verbosity_gate=1)

    def success(self, message: str):
        self._log(logging.INFO, message, event_type="success", verbosity_gate=0)

    def warning(self, message: str):
        self._log(logging.WARNING, message, event_type="warning", verbosity_gate=0)

    def error(self, message: str):
        self._log(logging.ERROR, message, event_type="error", verbosity_gate=0)

    def step(self, message: str, step: Optional[int] = None, total: Optional[int] = None):
        self._log(
            logging.INFO,
            message,
            event_type="step",
            verbosity_gate=0,
            step=step,
            total=total,
        )

    def section(self, message: str):
        self._log(logging.INFO, message, event_type="section", verbosity_gate=0)

    def blank(self):
        self._log(logging.INFO, "", event_type="blank", verbosity_gate=0)

    def progress_stage(
        self,
        message: str,
        *,
        step: int,
        total: int,
        elapsed: str | None = None,
        verbosity_gate: int = 0,
    ):
        self._log(
            logging.INFO,
            message,
            event_type="progress_stage",
            verbosity_gate=verbosity_gate,
            step=step,
            total=total,
            elapsed=elapsed,
        )

    def progress_update(self, message: str, *, elapsed: str, verbosity_gate: int = 1):
        self._log(
            logging.INFO,
            message,
            event_type="progress_update",
            verbosity_gate=verbosity_gate,
            elapsed=elapsed,
        )

    def progress_complete(
        self,
        message: str,
        *,
        elapsed: str,
        stage_time: float | None = None,
        verbosity_gate: int = 1,
    ):
        self._log(
            logging.INFO,
            message,
            event_type="progress_complete",
            verbosity_gate=verbosity_gate,
            elapsed=elapsed,
            stage_time=stage_time,
        )

    def module_progress(
        self,
        *,
        current: int,
        total: int,
        module_type: str,
        module_path: str,
        status: str,
        verbosity_gate: int = 2,
    ):
        self._log(
            logging.INFO,
            module_path,
            event_type="module_progress",
            verbosity_gate=verbosity_gate,
            current=current,
            total=total,
            module_type=module_type,
            module_path=module_path,
            status=status,
        )

    def elapsed_time(self) -> str:
        """Get elapsed time since logger was created."""
        elapsed = datetime.now() - self.start_time
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)

        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _log(self, level: int, message: str, **extra: Any):
        self.logger.log(level, message, extra=extra)


def create_logger(verbosity: int = 0, name: str = f"{LOGGER_NAME}.cli") -> CLILogger:
    """Create and return a configured CodeWiki logger wrapper."""
    return CLILogger(verbosity=verbosity, name=name)
