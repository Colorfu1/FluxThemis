from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from fluxthemis.structures import ActionChunk, TaskSpec


class BaseModelClient(ABC):
    """Policy boundary used by the evaluation runner.

    Implementations must ensure that preprocessing, inference and action
    denormalization happen locally or in their remote policy service. Returned
    actions must already be in environment units.
    """

    def setup(self) -> None:
        """Allocate model or network resources."""

    def reset(self, task: TaskSpec, episode_index: int, seed: int) -> None:
        """Reset episode-local model state."""

    @property
    def supports_explicit_episode_id(self) -> bool:
        """Whether the transport can use a coordinator-provided episode id."""
        return False

    def set_episode_id(self, episode_id: str) -> None:
        """Set the exact id used by subsequent episode inference requests."""
        raise NotImplementedError(
            f'{type(self).__name__} does not support explicit episode ids')

    @property
    def evaluation_reporting_enabled(self) -> bool:
        """Whether evaluation lifecycle events have an acknowledged sink."""
        return False

    @property
    def evaluation_report_run_id(self) -> str | None:
        """Opaque reporting-session identifier, when reporting is enabled."""
        return None

    @property
    def evaluation_report_run_dir(self) -> str | None:
        """Authoritative remote artifact directory acknowledged by the sink."""
        return None

    def report_evaluation(self, event_type: str,
                          payload: Mapping[str, Any]) -> str | None:
        """Report one lifecycle event and return its authoritative run dir.

        Local model clients intentionally implement this as a no-op. Runners
        must check :attr:`evaluation_reporting_enabled` before switching away
        from their local artifact path.
        """
        return None

    @abstractmethod
    def predict_action(self, observation: Any,
                       context: Mapping[str, Any]) -> ActionChunk:
        """Return float32 actions with shape ``[T, A]``."""

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
