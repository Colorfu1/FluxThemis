"""Exercise slot execution and worker capability messages over real ZMQ."""

import json
import threading
from pathlib import Path

import pytest

from fluxthemis.engines.runners import worker
from fluxthemis.engines.runners.coordinator import \
    DistributedEvaluationCoordinator
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings)
from fluxthemis.environments import MockVectorEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.reporters import MockEvaluationReporter


class _ExecutionEnvironment(MockVectorEnvironment):

    def __init__(self, directory, reject_reset, requires_single_task,
                 **kwargs):
        super().__init__(**kwargs)
        self.directory = Path(directory)
        self.reject_reset = reject_reset
        self.requires_single_task = requires_single_task
        for index, environment in enumerate(self.environments):
            environment.success_after = index * 2 + 1

    def reset_batch(self, requests):
        with (self.directory / 'resets.jsonl').open('a') as stream:
            stream.write(
                json.dumps([
                    None if request is None else
                    [request.episode_index, request.environment_case_index]
                    for request in requests
                ]) + '\n')
        results = super().reset_batch(requests)
        for request, result in zip(requests, results):
            if (request is not None and self.reject_reset
                    and request.environment_case_index == 0):
                result.rejected = True
                result.info['reason'] = 'bad initial layout'
        return results


def _mock_worker_process(**kwargs):
    """Supply CPU components while retaining the actual worker process loop."""
    config = json.loads(Path(kwargs['config_path']).read_text())
    evaluation = dict(
        episode_batch_size=config['capacity'],
        environment=dict(
            type=_ExecutionEnvironment,
            directory=config['directory'],
            reject_reset=config['reject_reset'],
            requires_single_task=config['requires_single_task'],
            num_envs=config['capacity']),
        policy_client=dict(type='MockPolicyClient', chunk_size=1),
        max_episode_steps=10)
    worker.load_distributed_evaluation_config = lambda *a, **kw: evaluation
    worker._run_simulator_worker(**kwargs)


@pytest.mark.parametrize('capacity', [1, 2, 3])
@pytest.mark.parametrize('reject_reset', [False, True])
@pytest.mark.parametrize('requires_single_task', [False, True])
def test_supervised_slot_execution(tmp_path, monkeypatch, capacity,
                                   reject_reset, requires_single_task):
    endpoint = f'ipc://{tmp_path}/workers.sock'
    config = tmp_path / 'config.json'
    config.write_text(
        json.dumps(
            dict(
                directory=str(tmp_path),
                capacity=capacity,
                reject_reset=reject_reset,
                requires_single_task=requires_single_task)))
    monkeypatch.setattr(worker, '_run_simulator_worker', _mock_worker_process)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    supervisor = worker.SimulatorWorkerSupervisor(
        str(config), endpoint, 'simulator-worker-0', 'deployment', '0', 10)
    errors = []

    def supervise():
        try:
            supervisor.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=supervise, daemon=True)
    coordinator = DistributedEvaluationCoordinator(
        endpoint,
        1,
        startup_timeout_s=10,
        episode_timeout_s=10,
        deployment_id='deployment')
    monkeypatch.setattr(coordinator, '_install_signal_handlers', lambda: None)
    reporter = MockEvaluationReporter()
    with EvaluationRun(
            EvaluationSettings(
                episodes_per_task=5, episode_batch_size=capacity),
            SuccessRateEvaluator(), reporter) as run:
        thread.start()
        try:
            summary = coordinator.run(run)
            assert summary['run']['completed_episodes'] == 5
        finally:
            thread.join(timeout=15)
    assert not thread.is_alive()
    assert not errors
    slot = coordinator._worker_slots['simulator-worker-0']
    assert slot.capacity == capacity
    assert slot.requires_single_task == requires_single_task
    ends = [
        payload for kind, payload in reporter.events if kind == 'episode_end'
    ]
    assert len({end['episode_id'] for end in ends}) == 5
    assert sorted(end['environment_case_index'] for end in ends) == (
        list(range(1, 6)) if reject_reset else list(range(5)))
    assert sorted(end['policy_seed'] for end in ends) == list(range(7, 12))
