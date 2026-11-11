"""Worker assignments and the supported execution/recovery policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .episode import EpisodeJob


@dataclass(frozen=True)
class EpisodeAssignment:
    """One ordered, same-task group occupying all slots in a worker."""

    jobs: tuple[EpisodeJob, ...]

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError('An assignment must contain at least one episode')
        if len({job.episode_id for job in self.jobs}) != len(self.jobs):
            raise ValueError('Assignment contains duplicate episode IDs')
        if any(job.task != self.jobs[0].task for job in self.jobs):
            raise ValueError('An assignment must contain one task only')

    @property
    def assignment_id(self) -> str:
        return self.jobs[0].episode_id


@dataclass(frozen=True)
class ExecutionPolicy:
    """First-version batch limitations, shared by execution and scheduling.

    Scalar episodes retain retries, case replacement, recovery and video.
    Native batches are atomic execution groups: acknowledged episode results
    survive a failure, but unfinished groups cannot be retried or resumed.
    """

    batch_size: int = 1

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('episode_batch_size must be a positive integer')

    def require_recovery(self, operation: str) -> None:
        if self.batch_size != 1:
            raise ValueError(f'Batch execution does not support {operation}; '
                             'completed results are retained')

    def validate_options(self,
                         *,
                         save_videos: bool = False,
                         resume: bool = False,
                         strict_worker_assignment: bool = False) -> None:
        if resume:
            self.require_recovery('resume')
        if self.batch_size != 1:
            if save_videos:
                raise ValueError('Batch execution does not yet support video')
            if strict_worker_assignment:
                raise ValueError(
                    'Batch execution requires dynamic scheduling; '
                    'fixed model-worker pairing is scalar-only')

    def validate_jobs(self, jobs: Sequence[EpisodeJob]) -> None:
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job.task.task_id] = counts.get(job.task.task_id, 0) + 1
        for task_id, count in counts.items():
            if count % self.batch_size:
                raise ValueError(f'Task {task_id!r} has {count} episodes; '
                                 'episode count must be divisible by '
                                 f'episode_batch_size={self.batch_size}')
