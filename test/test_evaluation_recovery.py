import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from mmengine import Config

from fluxthemis.engines.runners.assignment import EpisodeAssignment
from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                execute_assignment)
from fluxthemis.engines.runners.evaluation import (EvaluationRun,
                                                   EvaluationSettings, utc_now)
from fluxthemis.environments import MockEnvironment
from fluxthemis.evaluators import SuccessRateEvaluator
from fluxthemis.model_clients import MockPolicyClient
from fluxthemis.reporters import MockEvaluationReporter
from fluxthemis.structures import TaskSpec

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_retains_explicit_case_replacement(tmp_path):
    from fluxthemis.engines.runners.resume import (build_bundle,
                                                   validate_journal)

    reporter = MockEvaluationReporter()
    tasks = [TaskSpec('task')]
    job = build_episode_jobs(tasks, 1, 7, reporter.run_session_id)[0]
    replacement = replace(job, environment_case_index=1)
    with EvaluationRun(EvaluationSettings(), SuccessRateEvaluator(),
                       reporter) as run:
        run.start(tasks, 1)
        run.on_episode_start(job.episode_start_payload(utc_now()))
        payload = replacement.episode_start_payload(utc_now())
        payload['replaces_environment_case_index'] = 0
        run.on_episode_start(payload)
        environment = MockEnvironment(success_after=1)
        environment.setup()
        result = execute_assignment(environment, [MockPolicyClient()],
                                    EpisodeAssignment([replacement]), 5, None,
                                    True)[0]
        run.on_episode_end(result, replacement.episode_id)
        run.finish(result)
    rows = [
        dict(
            event_type=kind,
            payload=body,
            sequence=index,
            run_session_id=reporter.run_session_id,
            recorded_at=utc_now())
        for index, (kind, body) in enumerate(reporter.events, 1)
    ]
    journal = tmp_path / 'events.jsonl'
    journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    bundle = build_bundle(tmp_path)
    assert len(bundle['completed']) == 1
    assert bundle['completed'][0]['start']['environment_case_index'] == 1
    assert bundle['jobs'][0]['environment_case_index'] == 1
    rows[2]['payload'].pop('replaces_environment_case_index')
    with pytest.raises(ValueError, match='Duplicate episode_start'):
        validate_journal(rows)


def _config(capacity):
    evaluation = dict(
        environment=dict(type='MockVectorEnvironment', num_envs=capacity),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        episodes_per_task=2,
        max_episode_steps=5)
    if capacity != 1:
        evaluation['episode_batch_size'] = capacity
    return Config(
        dict(
            themis=dict(
                evaluation=evaluation,
                transport=dict(image_keys=['image'], state_keys=['state']))))


@pytest.mark.parametrize('stale_package', [False, True])
@pytest.mark.parametrize('dry_run', [False, True])
def test_launcher_imports_current_checkout(tmp_path, stale_package, dry_run):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    if stale_package:
        package = tmp_path / 'old_install' / 'fluxthemis'
        package.mkdir(parents=True)
        (package / '__init__.py'
         ).write_text('raise RuntimeError("Imported another checkout")\n')
        env['PYTHONPATH'] = str(package.parent)
    command = [sys.executable, str(ROOT / 'scripts/run_evaluation.py')]
    if dry_run:
        path = tmp_path / 'evaluation.py'
        _config(2).dump(str(path))
        pointer = tmp_path / 'result_pointer.json'
        command += [
            '--dry-run', '--config',
            str(path), '--gpu-ids', '0', '--server-endpoint',
            'tcp://127.0.0.1:15555', '--output',
            str(tmp_path / 'output'), '--result-pointer',
            str(pointer)
        ]
    else:
        command += ['--help']
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    if dry_run:
        output = Path(json.loads(pointer.read_text())['directory'])
        frozen = Config.fromfile(str(output / 'config_snapshot.py'))
        assert frozen.themis.evaluation.episode_batch_size == 2
        assert not (output / 'simulator.log').exists()
