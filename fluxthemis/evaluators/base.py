from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from fluxthemis.structures import EpisodeResult, StepResult, TaskSpec


class BaseEvaluator(ABC):
    """Lifecycle hooks for metrics and evaluation artifacts."""

    supports_parallel_rollouts = False

    def setup(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def on_run_start(self, tasks: Sequence[TaskSpec]) -> None:
        """Reset run-level metric state."""

    def on_episode_start(self, task: TaskSpec, episode_index: int,
                         seed: int) -> None:
        """Reset episode-level metric state."""

    def on_step(self, result: StepResult) -> None:
        """Observe one environment transition."""

    @abstractmethod
    def on_episode_end(self, result: EpisodeResult) -> None:
        """Consume a completed episode."""

    @abstractmethod
    def summarize(self) -> dict:
        """Return a JSON-serializable run summary."""

    def close(self) -> None:
        """Release evaluator resources."""
