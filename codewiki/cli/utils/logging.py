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
        if event_type in {"progress_stage", "stage_start"}:
            return (
                f"[{getattr(record, 'elapsed', self._format_clock(record))}] "
                f"Stage {getattr(record, 'step', '?')}/{getattr(record, 'total', '?')}  {message}"
            )
        if event_type in {"progress_update", "stage_update"}:
            return f"[{getattr(record, 'elapsed', self._format_clock(record))}]   {message}"
        if event_type in {"progress_complete", "stage_complete"}:
            duration_seconds = getattr(record, "stage_time", None)
            suffix = ""
            if duration_seconds is not None:
                suffix = f"  DONE {duration_seconds:.1f}s"
            return (
                f"[{getattr(record, 'elapsed', self._format_clock(record))}] "
                f"Stage {getattr(record, 'step', '?')}/{getattr(record, 'total', '?')}  "
                f"{message}{suffix}"
            )
        if event_type in {"module_progress", "module_event"}:
            prefix = self._format_counter(
                getattr(record, "current", None),
                getattr(record, "total", None),
            )
            depth = max(0, int(getattr(record, "depth", 0)))
            indent = "  " * depth
            module_kind = getattr(
                record,
                "module_kind",
                getattr(record, "module_type", "module"),
            )
            module_path = getattr(record, "module_path", message)
            status = str(getattr(record, "status", "info")).upper()
            duration_seconds = getattr(record, "duration_seconds", None)
            line = f"{prefix}{indent}{module_kind}  {module_path}  {status}"
            if duration_seconds is not None:
                line += f" {duration_seconds:.1f}s"
            return line
        if event_type == "cache":
            subject = getattr(record, "cache_subject", "artifact")
            target = getattr(record, "cache_target", message)
            return f"[{self._format_clock(record)}] CACHE  {subject}  {target}"
        if event_type == "failure":
            return f"[{self._format_clock(record)}] FAIL  {message}"
        if event_type == "llm_request":
            request_tokens = getattr(record, "llm_request_tokens", None)
            request_tokens_display = "unknown" if request_tokens is None else str(request_tokens)
            return (
                f"[{self._format_clock(record)}] "
                f"LLM REQ  {getattr(record, 'llm_prompt_type', 'llm')}  in={request_tokens_display}"
            )
        if event_type == "llm_response":
            duration_seconds = getattr(record, "llm_duration_seconds", None)
            response_tokens = getattr(record, "llm_response_tokens", None)
            response_tokens_per_second = getattr(record, "llm_response_tokens_per_second", None)
            duration_display = "unknown" if duration_seconds is None else f"{duration_seconds:.3f}s"
            response_tokens_display = "unknown" if response_tokens is None else str(response_tokens)
            response_speed_display = (
                "unknown"
                if response_tokens_per_second is None
                else f"{response_tokens_per_second:.3f}"
            )
            return (
                f"[{self._format_clock(record)}] "
                f"LLM RES  {getattr(record, 'llm_prompt_type', 'llm')}  "
                f"dur={duration_display}  out={response_tokens_display}  "
                f"tps={response_speed_display}"
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

    def _format_counter(self, current: Any, total: Any) -> str:
        if not isinstance(current, int) or not isinstance(total, int):
            return ""
        width = max(2, len(str(current)), len(str(total)))
        return f"[{current:0{width}d}/{total:0{width}d}] "


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
        self.stage_start(
            message,
            step=step,
            total=total,
            elapsed=elapsed,
            verbosity_gate=verbosity_gate,
        )

    def stage_start(
        self,
        message: str,
        *,
        step: int,
        total: int,
        elapsed: str | None = None,
        verbosity_gate: int = 0,
    ):
        self._last_stage_step = step
        self._last_stage_total = total
        self._log(
            logging.INFO,
            message,
            event_type="stage_start",
            verbosity_gate=verbosity_gate,
            step=step,
            total=total,
            elapsed=elapsed,
        )

    def progress_update(self, message: str, *, elapsed: str, verbosity_gate: int = 1):
        self.stage_update(message, elapsed=elapsed, verbosity_gate=verbosity_gate)

    def stage_update(self, message: str, *, elapsed: str, verbosity_gate: int = 1):
        self._log(
            logging.INFO,
            message,
            event_type="stage_update",
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
        self.stage_complete(
            message,
            step=getattr(self, "_last_stage_step", None),
            total=getattr(self, "_last_stage_total", None),
            elapsed=elapsed,
            stage_time=stage_time,
            verbosity_gate=verbosity_gate,
        )

    def stage_complete(
        self,
        message: str,
        *,
        step: int | None,
        total: int | None,
        elapsed: str,
        stage_time: float | None = None,
        verbosity_gate: int = 1,
    ):
        self._log(
            logging.INFO,
            message,
            event_type="stage_complete",
            verbosity_gate=verbosity_gate,
            elapsed=elapsed,
            stage_time=stage_time,
            step=step,
            total=total,
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
        self.module_event(
            current=current,
            total=total,
            module_kind=module_type,
            module_path=module_path,
            status=status,
            verbosity_gate=verbosity_gate,
        )

    def module_event(
        self,
        *,
        current: int | None = None,
        total: int | None = None,
        module_kind: str,
        module_path: str,
        status: str,
        duration_seconds: float | None = None,
        depth: int = 0,
        verbosity_gate: int = 2,
    ):
        self._log(
            logging.INFO,
            module_path,
            event_type="module_event",
            verbosity_gate=verbosity_gate,
            current=current,
            total=total,
            module_kind=module_kind,
            module_path=module_path,
            status=status,
            duration_seconds=duration_seconds,
            depth=depth,
        )

    def cache(self, subject: str, target: str, *, verbosity_gate: int = 1):
        self._log(
            logging.INFO,
            target,
            event_type="cache",
            verbosity_gate=verbosity_gate,
            cache_subject=subject,
            cache_target=target,
        )

    def failure(self, message: str, *, verbosity_gate: int = 1):
        self._log(
            logging.ERROR,
            message,
            event_type="failure",
            verbosity_gate=verbosity_gate,
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


def log_module_event(
    logger: logging.Logger,
    *,
    module_kind: str,
    module_path: str,
    status: str,
    current: int | None = None,
    total: int | None = None,
    duration_seconds: float | None = None,
    depth: int = 0,
    verbosity_gate: int = 2,
) -> None:
    """Emit a structured module lifecycle event."""
    logger.info(
        module_path,
        extra={
            "event_type": "module_event",
            "verbosity_gate": verbosity_gate,
            "module_kind": module_kind,
            "module_path": module_path,
            "status": status,
            "current": current,
            "total": total,
            "duration_seconds": duration_seconds,
            "depth": depth,
        },
    )


def log_cache_event(
    logger: logging.Logger,
    *,
    subject: str,
    target: str,
    verbosity_gate: int = 1,
) -> None:
    """Emit a structured cache notice."""
    logger.info(
        target,
        extra={
            "event_type": "cache",
            "verbosity_gate": verbosity_gate,
            "cache_subject": subject,
            "cache_target": target,
        },
    )


def log_failure_event(
    logger: logging.Logger,
    message: str,
    *,
    verbosity_gate: int = 1,
) -> None:
    """Emit a short structured failure event."""
    logger.error(
        message,
        extra={
            "event_type": "failure",
            "verbosity_gate": verbosity_gate,
        },
    )
