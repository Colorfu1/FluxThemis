import pytest

from fluxthemis.engines.runners.assignment import EpisodeAssignment
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_assignment)
from fluxthemis.environments.mock import MockVectorEnvironment
from fluxthemis.model_clients.mock import MockPolicyClient
from fluxthemis.structures import TaskSpec


@pytest.mark.parametrize('capacity', [1, 3])
def test_episode_jobs_and_execution_contract(capacity):
    jobs = build_episode_jobs([TaskSpec('a'), TaskSpec('b')], 2, 7, 'run')
    assert [job.policy_seed for job in jobs] == [7, 8, 9, 10]
    assert [job.environment_seed for job in jobs] == [7, 8, 9, 10]
    assert [job.environment_case_index for job in jobs] == [0, 1, 0, 1]
    environment = MockVectorEnvironment(capacity, success_after=2)
    environment.setup()
    results = execute_assignment(environment, [
        MockPolicyClient(action_dim=7, chunk_size=1) for _ in range(capacity)
    ], EpisodeAssignment(jobs[:capacity]), 3, 1, True)
    for job, result in zip(jobs, results):
        job.validate_result(result)
        assert result.success and result.steps == 2
