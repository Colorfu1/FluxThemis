import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / 'scripts'


@pytest.mark.parametrize(
    'name',
    [
        'install.sh',
        'install_ros1_hook.sh',
        'run_ros1_eval.sh',
    ],
)
def test_ros1_launch_scripts_have_valid_bash_syntax(name):
    subprocess.run(
        ['/bin/bash', '-n', str(SCRIPTS / name)],
        check=True,
    )


def test_one_command_eval_help_is_environment_independent():
    completed = subprocess.run(
        ['/bin/bash', str(SCRIPTS / 'run_ros1_eval.sh'), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--smoke' in completed.stdout
    assert '--service-timeout' in completed.stdout
    assert '--shutdown-master' in completed.stdout
    assert '--workers' in completed.stdout
    assert '--sim-gpus' in completed.stdout
    assert 'FLUXTHEMIS_PARALLEL_WORKERS' in completed.stdout
    assert 'FLUXTHEMIS_SIM_GPU_IDS' in completed.stdout
    assert '--service-name' not in completed.stdout
    assert 'runner-only --cfg-options' in completed.stdout
    assert 'fluxthemis-eval' in completed.stdout


def test_installer_documents_fluxvla_ros1_hook():
    completed = subprocess.run(
        ['/bin/bash', str(SCRIPTS / 'install.sh'), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--fluxvla-env-name' in completed.stdout
    assert '--skip-fluxvla-hook' in completed.stdout
