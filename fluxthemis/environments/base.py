from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from fluxthemis.structures import StepResult, TaskSpec


class BaseEnvironment(ABC):
    """Contract implemented by simulation and real-robot adapters."""

    def setup(self) -> None:
        """Allocate simulator or hardware resources."""

    @abstractmethod
    def tasks(self) -> Sequence[TaskSpec]:
        """Return the tasks available for evaluation."""

    @abstractmethod
    def reset(self, task: TaskSpec, episode_index: int,
              seed: int) -> tuple[Any, dict]:
        """Reset one episode and return ``(observation, info)``."""

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """Execute one environment-unit action."""

    def render(self) -> Any | None:
        return None

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
