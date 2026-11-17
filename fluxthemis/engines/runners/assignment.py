"""Worker assignments and environment execution capabilities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from fluxthemis.model_clients.base import BasePolicyClient

    from .episode import EpisodeJob


@dataclass(frozen=True)
class EpisodeAssignment:
    """An ordered list of distinct episodes assigned to one worker."""

    jobs: list[EpisodeJob]
    assignment_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'jobs', list(self.jobs))
        if not self.jobs:
            raise ValueError('An assignment must contain at least one episode')
        if len({job.episode_id for job in self.jobs}) != len(self.jobs):
            raise ValueError('Assignment contains duplicate episode IDs')
        if not isinstance(self.assignment_id, str) or not self.assignment_id:
            raise ValueError('Assignment ID must be a nonempty string')


@dataclass(frozen=True)
class ExecutionPolicy:
    """Validate capacity and the capabilities declared by an environment."""

    batch_size: int = 1
    requires_single_task: bool = False

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('episode_batch_size must be a positive integer')
        if type(self.requires_single_task) is not bool:
            raise TypeError('requires_single_task must be a boolean')

    def validate_capacity(self, capacity: int) -> None:
        if type(capacity) is not int or capacity != self.batch_size:
            raise ValueError(
                'Environment capacity differs from episode_batch_size')

    def validate_assignment(
            self,
            assignment: EpisodeAssignment,
            policy_clients: Sequence[BasePolicyClient] | None = None) -> None:
        if len(assignment.jobs) > self.batch_size:
            raise ValueError('Assignment exceeds worker capacity')
        if self.requires_single_task and any(
                job.task != assignment.jobs[0].task
                for job in assignment.jobs):
            raise ValueError('Environment requires one task per assignment')
        if policy_clients is not None:
            if len(policy_clients) != self.batch_size:
                raise ValueError('Environment and policy client sizes differ')
            if len({id(client)
                    for client in policy_clients}) != len(policy_clients):
                raise ValueError(
                    'Each environment slot needs its own policy client')
