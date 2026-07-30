import fluxthemis  # noqa: F401
from fluxthemis.engines import (ENVIRONMENTS, EVALUATORS, MODEL_CLIENTS,
                                RUNNERS, build_runner_from_cfg)


def test_builtin_components_are_registered():
    assert ENVIRONMENTS.get('MockEnvironment') is not None
    assert ENVIRONMENTS.get('GymnasiumEnvironment') is not None
    assert MODEL_CLIENTS.get('MockModelClient') is not None
    assert MODEL_CLIENTS.get('FluxVLAROSModelClient') is not None
    assert MODEL_CLIENTS.get('FluxVLAROS2ModelClient') is not None
    assert EVALUATORS.get('SuccessRateEvaluator') is not None
    assert RUNNERS.get('EvalRunner') is not None


def test_nested_components_build_from_fluxvla_style_config(tmp_path):
    runner = build_runner_from_cfg(
        dict(
            type='EvalRunner',
            environment=dict(type='MockEnvironment'),
            model_client=dict(type='MockModelClient'),
            evaluator=dict(type='SuccessRateEvaluator'),
            work_dir=str(tmp_path),
            run_name='registry',
        ))

    assert type(runner).__name__ == 'EvalRunner'
    assert type(runner.environment).__name__ == 'MockEnvironment'
    assert type(runner.model_client).__name__ == 'MockModelClient'
    assert type(runner.evaluator).__name__ == 'SuccessRateEvaluator'
