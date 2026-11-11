"""Exercise process supervision and coordinator decisions over real ZMQ."""

import json
import os
import threading
import time
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


class _CrashEnvironment(MockVectorEnvironment):

    def __init__(self, directory, mode, exitcode, **kwargs):
        super().__init__(**kwargs)
        self.directory = Path(directory)
        self.mode = mode
        self.exitcode = exitcode
        if self.num_envs > 1:
            self.environments[1].success_after = 3

    def reset_batch(self, requests):
        self.episode_index = requests[0].episode_index
        return super().reset_batch(requests)

    def step_batch(self, actions):
        if self.mode == 'scalar_retry':
            marker = self.directory / f'crashed-{self.episode_index}'
            if not marker.exists():
                marker.touch()
                os._exit(9)
        elif self.mode == 'batch_failure' and actions[0] is None:
            # Crash only after the coordinator acknowledges the first result.
            deadline = time.monotonic() + 5
            while not (self.directory / 'reported').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('First episode was not reported')
                time.sleep(0.01)
            os._exit(self.exitcode)
        return super().step_batch(actions)


def _mock_worker_process(**kwargs):
    """Supply CPU components while retaining the real worker process loop."""
    config = json.loads(Path(kwargs['config_path']).read_text())
    directory = Path(config['directory'])
    with (directory / 'generations').open('a') as stream:
        stream.write(f'{kwargs["generation"]}\n')
    evaluation = dict(
        episode_batch_size=config['capacity'],
        environment=dict(
            type=_CrashEnvironment,
            directory=str(directory),
            mode=config['mode'],
            exitcode=config['exitcode'],
            num_envs=config['capacity'],
            success_after=1),
        policy_client=dict(type='MockPolicyClient', chunk_size=1),
        max_episode_steps=5)
    worker.load_distributed_evaluation_config = lambda *a, **kw: evaluation
    # Spawn imports this module afresh, so the production target is unchanged
    # here even though the parent substitutes this wrapper as its spawn target.
    worker._run_simulator_worker(**kwargs)


class _Reporter(MockEvaluationReporter):

    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def report_evaluation(self, event_type, payload):
        result = super().report_evaluation(event_type, payload)
        if event_type == 'episode_end':
            (self.directory / 'reported').touch()
        return result


@pytest.mark.parametrize('mode,capacity,episodes,generations,exitcode', [
    ('batch_complete', 2, 4, [0], 9),
    ('batch_failure', 2, 2, [0], 9),
    ('batch_failure', 2, 2, [0], 0),
    ('scalar_retry', 1, 4, [0, 1, 2, 3, 4], 9),
])
def test_supervised_evaluation(tmp_path, monkeypatch, mode, capacity, episodes,
                               generations, exitcode):
    endpoint = f'ipc://{tmp_path}/workers.sock'
    config = tmp_path / 'config.json'
    config.write_text(
        json.dumps(
            dict(
                directory=str(tmp_path),
                mode=mode,
                capacity=capacity,
                exitcode=exitcode)))
    monkeypatch.setattr(worker, '_run_simulator_worker', _mock_worker_process)
    monkeypatch.setattr(worker, '_SIMULATOR_STARTUP_TIMEOUT_S', 10)
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
        simulator_restart_limit=1,
        deployment_id='deployment')
    monkeypatch.setattr(coordinator, '_install_signal_handlers', lambda: None)
    reporter = _Reporter(tmp_path)
    run = EvaluationRun(
        EvaluationSettings(
            episodes_per_task=episodes, episode_batch_size=capacity),
        SuccessRateEvaluator(), reporter)
    thread.start()
    try:
        with run:
            if mode == 'batch_failure':
                with pytest.raises(ValueError, match='worker restart'):
                    coordinator.run(run)
            else:
                summary = coordinator.run(run)
                assert summary['run']['completed_episodes'] == episodes
    finally:
        thread.join(timeout=15)
    assert not thread.is_alive()
    assert list(map(int,
                    (tmp_path /
                     'generations').read_text().splitlines())) == generations
    end = reporter.events[-1]
    assert end[0] == 'run_end'
    if mode == 'batch_failure':
        assert len(errors) == 1
        assert f'stopped after exit code {exitcode}' in str(errors[0])
        assert end[1]['status'] == 'failed'
        assert end[1]['completed_episodes'] == 1
    else:
        assert not errors
        assert end[1]['status'] == 'finished'
    if mode == 'scalar_retry':
        state = coordinator._scheduling_state
        assert list(state.restart_counts_by_episode_id.values()) == [1] * 4


def test_startup_failure_stops_before_exceeding_restart_budget(
        tmp_path, monkeypatch):
    # A missing config fails before readiness; use the same real supervisor
    # handshake to check the independent startup retry budget.
    endpoint = f'ipc://{tmp_path}/workers.sock'
    coordinator = DistributedEvaluationCoordinator(
        endpoint,
        1,
        startup_timeout_s=10,
        simulator_restart_limit=0,
        deployment_id='deployment')
    monkeypatch.setattr(coordinator, '_install_signal_handlers', lambda: None)
    monkeypatch.setattr(worker, '_SIMULATOR_STARTUP_TIMEOUT_S', 10)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    errors = []

    def supervise():
        try:
            worker.SimulatorWorkerSupervisor(
                str(tmp_path / 'missing.py'), endpoint, 'simulator-worker-0',
                'deployment', '0', 10).run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=supervise, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match='startup restart limit'):
            coordinator.run(None)
    finally:
        thread.join(timeout=15)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert coordinator._worker_slots['simulator-worker-0'].generation == 0
