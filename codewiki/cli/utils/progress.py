"""
Progress indicator utilities for CLI.
"""

from __future__ import annotations

import time
from typing import Optional

from codewiki.cli.utils.logging import CLILogger, create_logger, normalize_verbosity


class ProgressTracker:
    """
    Progress tracker with stages and ETA estimation.

    Stages:
    1. Dependency Analysis (40% of time)
    2. Module Clustering (20% of time)
    3. Documentation Generation (30% of time)
    4. HTML Generation (5% of time, optional)
    5. Finalization (5% of time)
    """

    STAGE_WEIGHTS = {
        1: 0.40,
        2: 0.20,
        3: 0.30,
        4: 0.05,
        5: 0.05,
    }

    STAGE_NAMES = {
        1: "Dependency Analysis",
        2: "Module Clustering",
        3: "Documentation Generation",
        4: "HTML Generation",
        5: "Finalization",
    }

    def __init__(
        self,
        total_stages: int = 5,
        verbosity: int = 0,
        logger: CLILogger | None = None,
    ):
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_progress = 0.0
        self.start_time = time.time()
        self.verbosity = normalize_verbosity(verbosity)
        self.current_stage_start = self.start_time
        self.logger = logger or create_logger(
            verbosity=self.verbosity, name="codewiki.cli.progress"
        )

    def start_stage(self, stage: int, description: Optional[str] = None):
        self.current_stage = stage
        self.stage_progress = 0.0
        self.current_stage_start = time.time()

        stage_name = description or self.STAGE_NAMES.get(stage, f"Stage {stage}")
        elapsed = self._format_elapsed()
        gate = 1 if self.verbosity >= 1 else 0
        self.logger.progress_stage(
            stage_name,
            step=stage,
            total=self.total_stages,
            elapsed=elapsed,
            verbosity_gate=gate,
        )

    def update_stage(self, progress: float, message: Optional[str] = None):
        self.stage_progress = min(1.0, max(0.0, progress))

        if self.verbosity >= 1 and message:
            self.logger.progress_update(message, elapsed=self._format_elapsed(), verbosity_gate=1)

    def complete_stage(self, message: Optional[str] = None):
        self.stage_progress = 1.0

        if self.verbosity >= 1:
            stage_time = time.time() - self.current_stage_start
            stage_name = self.STAGE_NAMES.get(self.current_stage, f"Stage {self.current_stage}")
            self.logger.progress_complete(
                stage_name + " complete",
                elapsed=self._format_elapsed(),
                stage_time=stage_time,
                verbosity_gate=1,
            )
            if message:
                self.logger.progress_update(
                    message, elapsed=self._format_elapsed(), verbosity_gate=1
                )

    def detail(self, message: str, min_verbosity: int = 2):
        if self.verbosity >= min_verbosity:
            self.logger.progress_update(
                message,
                elapsed=self._format_elapsed(),
                verbosity_gate=min_verbosity,
            )

    def get_overall_progress(self) -> float:
        completed_weight = sum(self.STAGE_WEIGHTS.get(s, 0) for s in range(1, self.current_stage))
        current_weight = self.STAGE_WEIGHTS.get(self.current_stage, 0) * self.stage_progress
        return completed_weight + current_weight

    def _format_elapsed(self) -> str:
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        if minutes > 0:
            return f"{minutes:02d}:{seconds:02d}"
        return f"00:{seconds:02d}"

    def get_eta(self) -> Optional[str]:
        elapsed = time.time() - self.start_time
        progress = self.get_overall_progress()

        if progress <= 0.0:
            return None

        total_estimated = elapsed / progress
        remaining = total_estimated - elapsed

        if remaining < 0:
            return "< 1 min"

        minutes = int(remaining // 60)
        seconds = int(remaining % 60)

        if minutes > 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


class ModuleProgressBar:
    """Deterministic module progress reporter."""

    def __init__(
        self,
        total_modules: int,
        verbosity: int = 0,
        logger: CLILogger | None = None,
    ):
        self.total_modules = total_modules
        self.current_module = 0
        self.verbosity = normalize_verbosity(verbosity)
        self.logger = logger or create_logger(
            verbosity=self.verbosity, name="codewiki.cli.module_progress"
        )

    def update(
        self,
        module_name: str,
        cached: bool = False,
        *,
        module_type: str | None = None,
        module_path: str | None = None,
        status: str | None = None,
    ):
        self.current_module += 1
        resolved_status = status or ("cached" if cached else "generated")

        if self.verbosity >= 2:
            self.logger.module_progress(
                current=self.current_module,
                total=self.total_modules,
                module_type=module_type or "module",
                module_path=module_path or module_name,
                status=resolved_status,
                verbosity_gate=2,
            )
        elif self.verbosity >= 1:
            display_status = "✓ (cached)" if cached else "⟳ (generating)"
            self.logger.module_progress(
                current=self.current_module,
                total=self.total_modules,
                module_type="module",
                module_path=f"{module_name}... {display_status}",
                status="",
                verbosity_gate=1,
            )

    def finish(self):
        """Keep compatibility with the previous API."""
