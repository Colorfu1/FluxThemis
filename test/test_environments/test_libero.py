from types import SimpleNamespace

import numpy as np
import pytest

from fluxthemis.environments.sim import libero as libero_module
from fluxthemis.environments.sim.libero import (LIBERO_ACTION_DIM,
                                                LIBERO_DUMMY_ACTION,
                                                LiberoEnvironment)
from fluxthemis.structures import StepResult, TaskSpec


def _observation(marker):
    base_image = (
        np.arange(18, dtype=np.uint16).reshape(2, 3, 3) + marker * 20)
    base_image = np.asarray(base_image % 256, dtype=np.uint8)
    wrist_image = np.asarray(
        (base_image.astype(np.uint16) + 7) % 256, dtype=np.uint8)
    return {
        'agentview_image':
        base_image,
        'robot0_eye_in_hand_image':
        wrist_image,
        'robot0_eef_pos':
        np.asarray([marker, marker + 1, marker + 2], dtype=np.float64),
        'robot0_eef_quat':
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        'robot0_gripper_qpos':
        np.asarray([marker, -marker], dtype=np.float32),
        'untouched_metadata': {
            'marker': marker
        },
    }


class _FakeLiberoEnv:

    def __init__(self, kwargs, step_results=None, seed_error=None):
        self.kwargs = dict(kwargs)
        self.step_results = list(step_results or [])
        self.seed_error = seed_error
        self.seed_calls = []
        self.reset_calls = 0
        self.init_state_calls = []
        self.step_actions = []
        self.step_observations = []
        self.close_calls = 0
        self.events = []

    def seed(self, seed):
        self.events.append(('seed', seed))
        self.seed_calls.append(seed)
        if self.seed_error is not None:
            raise self.seed_error

    def reset(self):
        self.events.append(('reset', None))
        self.reset_calls += 1
        return _observation(-1)

    def set_init_state(self, state):
        self.events.append(('set_init_state', state))
        self.init_state_calls.append(state)
        return _observation(0)

    def step(self, action):
        action = list(action)
        self.events.append(('step', action))
        self.step_actions.append(action)
        if self.step_results:
            result = self.step_results.pop(0)
        else:
            marker = len(self.step_actions)
            result = (_observation(marker), 0.0, False, {'step': marker})
        self.step_observations.append(result[0])
        return result

    def close(self):
        self.events.append(('close', None))
        self.close_calls += 1


class _FakeEnvModule:

    def __init__(self):
        self.instances = []
        self.next_step_results = []
        self.next_seed_error = None

    def queue_step_results(self, *results):
        self.next_step_results.append(list(results))

    def OffScreenRenderEnv(self, **kwargs):
        step_results = (
            self.next_step_results.pop(0) if self.next_step_results else None)
        env = _FakeLiberoEnv(
            kwargs,
            step_results=step_results,
            seed_error=self.next_seed_error,
        )
        self.next_seed_error = None
        self.instances.append(env)
        return env


class _FakeSuite:

    def __init__(self):
        self.task_values = [
            SimpleNamespace(
                name='pick_up_the_black_bowl',
                language='pick up the black bowl',
                problem_folder='libero_10',
                bddl_file='pick_up_the_black_bowl.bddl',
            ),
            SimpleNamespace(
                name='open_the_middle_drawer',
                language='open the middle drawer',
                problem_folder='libero_10',
                bddl_file='open_the_middle_drawer.bddl',
            ),
        ]
        self.initial_states = {
            0: ('task-0-state-0', 'task-0-state-1'),
            1: ('task-1-state-0', ),
        }
        self.init_state_requests = []

    @property
    def n_tasks(self):
        return len(self.task_values)

    def get_task(self, task_index):
        return self.task_values[task_index]

    def get_task_init_states(self, task_index):
        self.init_state_requests.append(task_index)
        return self.initial_states[task_index]


class _FakeBenchmark:

    def __init__(self, suite):
        self.suite = suite
        self.factory_calls = []
        self.available = True

    def get_benchmark_dict(self):
        if not self.available:
            return {'libero_goal': object()}

        def build_suite(task_order_index):
            self.factory_calls.append(task_order_index)
            return self.suite

        return {'libero_10': build_suite}


class _FakeLiberoModule:

    def __init__(self):
        self.path_requests = []

    def get_libero_path(self, key):
        self.path_requests.append(key)
        assert key == 'bddl_files'
        return '/fake/libero/bddl'


@pytest.fixture
def fake_stack(monkeypatch):
    suite = _FakeSuite()
    benchmark = _FakeBenchmark(suite)
    envs = _FakeEnvModule()
    libero = _FakeLiberoModule()
    runtime = libero_module._LiberoRuntime(
        benchmark=benchmark,
        libero=libero,
        envs=envs,
    )
    monkeypatch.setattr(libero_module, '_load_libero_runtime', lambda: runtime)
    return SimpleNamespace(
        suite=suite,
        benchmark=benchmark,
        envs=envs,
        libero=libero,
        runtime=runtime,
    )


def _setup_environment(fake_stack, **kwargs):
    environment = LiberoEnvironment(
        task_suite_name='libero_10',
        **kwargs,
    )
    environment.setup()
    return environment


def _assert_observation_unchanged(actual, source):
    assert actual.keys() == source.keys()
    for key in (
            'agentview_image',
            'robot0_eye_in_hand_image',
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
    ):
        assert actual[key] is source[key]
        np.testing.assert_array_equal(actual[key], source[key])
        assert actual[key].dtype == source[key].dtype
        assert actual[key].shape == source[key].shape
    assert actual['untouched_metadata'] is source['untouched_metadata']


def test_setup_discovers_ordered_suite_tasks_and_is_idempotent(fake_stack):
    environment = LiberoEnvironment(
        task_suite_name='libero_10',
        task_order_index=3,
    )

    with pytest.raises(RuntimeError, match=r'setup\(\) was not called'):
        environment.tasks()

    environment.setup()
    environment.setup()
    tasks = environment.tasks()

    assert fake_stack.benchmark.factory_calls == [3]
    assert isinstance(tasks, tuple)
    assert [task.task_id for task in tasks] == ['0', '1']
    assert [task.description for task in tasks] == [
        'pick up the black bowl',
        'open the middle drawer',
    ]
    assert tasks[0].metadata == {
        'benchmark': 'libero',
        'task_suite_name': 'libero_10',
        'task_index': 0,
        'task_name': 'pick_up_the_black_bowl',
        'recommended_max_episode_steps': 520,
    }
    assert tasks[1].metadata['task_index'] == 1
    assert environment.recommended_max_episode_steps == 520
    assert fake_stack.envs.instances == []


def test_setup_rejects_suite_missing_from_installed_benchmark(fake_stack):
    fake_stack.benchmark.available = False
    environment = LiberoEnvironment(task_suite_name='libero_10')

    with pytest.raises(KeyError, match='available suites'):
        environment.setup()

    with pytest.raises(RuntimeError, match=r'setup\(\) was not called'):
        environment.tasks()


def test_reset_runs_warmup_and_keeps_raw_observation(fake_stack):
    environment = _setup_environment(
        fake_stack,
        task_order_index=2,
        resolution=128,
        controller='OSC_POSE',
        simulator_seed=0,
        num_steps_wait=10,
        env_kwargs={'render_gpu_device_id': 2},
    )
    task = environment.tasks()[0]

    observation, info = environment.reset(task, episode_index=3, seed=97)

    assert len(fake_stack.envs.instances) == 1
    simulator = fake_stack.envs.instances[0]
    assert simulator.kwargs == {
        'bddl_file_name':
        '/fake/libero/bddl/libero_10/pick_up_the_black_bowl.bddl',
        'camera_heights': 128,
        'camera_widths': 128,
        'controller': 'OSC_POSE',
        'render_gpu_device_id': 2,
    }
    assert simulator.seed_calls == [0]
    assert simulator.reset_calls == 1
    assert simulator.init_state_calls == ['task-0-state-1']
    assert simulator.events[:3] == [
        ('seed', 0),
        ('reset', None),
        ('set_init_state', 'task-0-state-1'),
    ]
    assert len(simulator.step_actions) == 10
    assert simulator.step_actions == [list(LIBERO_DUMMY_ACTION)] * 10
    assert fake_stack.suite.init_state_requests == [0]
    assert fake_stack.libero.path_requests == ['bddl_files']

    raw_final_observation = simulator.step_observations[-1]
    assert observation is not raw_final_observation
    _assert_observation_unchanged(observation, raw_final_observation)
    assert observation['agentview_image'][0, 0, 0] == \
        raw_final_observation['agentview_image'][0, 0, 0]
    assert 'task_description' not in observation
    assert 'is_new_episode' not in observation

    assert info == {
        'benchmark': 'libero',
        'task_suite_name': 'libero_10',
        'task_index': 0,
        'task_name': 'pick_up_the_black_bowl',
        'episode_seed': 97,
        'simulator_seed': 0,
        'init_state_index': 1,
        'warmup_steps': 10,
        'warmup_reward': 0.0,
        'warmup_success': False,
        'warmup_info': {
            'step': 10
        },
    }


def test_reset_rejects_unknown_task(fake_stack):
    environment = _setup_environment(fake_stack)

    with pytest.raises(KeyError, match='Unknown LIBERO task id'):
        environment.reset(TaskSpec('missing'), episode_index=0, seed=7)

    assert fake_stack.envs.instances == []


@pytest.mark.parametrize(
    'action,exception,message',
    [
        (np.asarray(0.0), ValueError, 'shape'),
        (np.zeros((1, LIBERO_ACTION_DIM)), ValueError, 'shape'),
        (np.zeros(LIBERO_ACTION_DIM - 1), ValueError, 'shape'),
        (['not-numeric'] * LIBERO_ACTION_DIM, TypeError, 'numeric'),
        ([False] * LIBERO_ACTION_DIM, TypeError, 'numeric'),
        ([0.0] * (LIBERO_ACTION_DIM - 1) + [np.nan], ValueError, 'NaN'),
        ([0.0] * (LIBERO_ACTION_DIM - 1) + [np.inf], ValueError, 'infinity'),
    ],
)
def test_step_rejects_invalid_actions_before_calling_simulator(
        fake_stack, action, exception, message):
    environment = _setup_environment(fake_stack, num_steps_wait=0)
    environment.reset(environment.tasks()[0], episode_index=0, seed=7)
    simulator = fake_stack.envs.instances[0]

    with pytest.raises(exception, match=message):
        environment.step(action)

    assert simulator.step_actions == []


def test_step_requires_reset(fake_stack):
    environment = _setup_environment(fake_stack, num_steps_wait=0)

    with pytest.raises(RuntimeError, match=r'reset\(\) must be called'):
        environment.step(np.zeros(LIBERO_ACTION_DIM, dtype=np.float32))


def test_step_forwards_float_action_and_maps_libero_success(fake_stack):
    running_observation = _observation(21)
    success_observation = _observation(22)
    fake_stack.envs.queue_step_results(
        (running_observation, 0.25, False, {
            'phase': 'running'
        }),
        (success_observation, 1.0, True, {
            'phase': 'success'
        }),
    )
    environment = _setup_environment(fake_stack, num_steps_wait=0)
    environment.reset(environment.tasks()[0], episode_index=0, seed=7)
    action = np.linspace(-1.0, 1.0, LIBERO_ACTION_DIM, dtype=np.float64)

    running = environment.step(action)
    success = environment.step(action)

    assert isinstance(running, StepResult)
    assert running.reward == 0.25
    assert running.terminated is False
    assert running.truncated is False
    assert running.success is False
    assert running.info == {
        'phase': 'running',
        'benchmark': 'libero',
        'task_suite_name': 'libero_10',
        'task_index': 0,
    }
    _assert_observation_unchanged(running.observation, running_observation)

    assert success.reward == 1.0
    assert success.terminated is True
    assert success.truncated is False
    assert success.success is True
    assert success.info['phase'] == 'success'
    _assert_observation_unchanged(success.observation, success_observation)

    simulator = fake_stack.envs.instances[0]
    assert len(simulator.step_actions) == 2
    for forwarded in simulator.step_actions:
        np.testing.assert_allclose(
            forwarded, action.astype(np.float32), rtol=0.0, atol=0.0)

    with pytest.raises(RuntimeError, match='already terminated'):
        environment.step(action)
    assert len(simulator.step_actions) == 2


def test_success_during_warmup_becomes_terminal_first_transition(fake_stack):
    warmup_observation = _observation(31)
    fake_stack.envs.queue_step_results((warmup_observation, 1.0, True, {
        'phase': 'warmup-success'
    }), )
    environment = _setup_environment(fake_stack, num_steps_wait=10)

    observation, info = environment.reset(
        environment.tasks()[0], episode_index=0, seed=7)
    transition = environment.step(np.zeros(LIBERO_ACTION_DIM))

    simulator = fake_stack.envs.instances[0]
    assert len(simulator.step_actions) == 1
    _assert_observation_unchanged(observation, warmup_observation)
    assert info['warmup_steps'] == 1
    assert info['warmup_reward'] == 1.0
    assert info['warmup_success'] is True
    assert info['warmup_info'] == {'phase': 'warmup-success'}
    assert transition.terminated is True
    assert transition.success is True
    assert transition.reward == 0.0
    assert transition.info == {
        'benchmark': 'libero',
        'success_during_warmup': True,
    }
    assert len(simulator.step_actions) == 1


def test_environment_reuses_same_task_and_closes_on_task_switch(fake_stack):
    environment = _setup_environment(
        fake_stack, simulator_seed=11, num_steps_wait=0)
    task_0, task_1 = environment.tasks()

    environment.reset(task_0, episode_index=0, seed=101)
    first = fake_stack.envs.instances[0]
    environment.reset(task_0, episode_index=3, seed=102)

    assert len(fake_stack.envs.instances) == 1
    assert first.seed_calls == [11]
    assert first.reset_calls == 2
    assert first.init_state_calls == ['task-0-state-0', 'task-0-state-1']
    assert first.close_calls == 0
    assert fake_stack.suite.init_state_requests == [0]

    environment.reset(task_1, episode_index=0, seed=103)
    second = fake_stack.envs.instances[1]

    assert len(fake_stack.envs.instances) == 2
    assert first.close_calls == 1
    assert second.seed_calls == [11]
    assert second.reset_calls == 1
    assert second.init_state_calls == ['task-1-state-0']
    assert fake_stack.suite.init_state_requests == [0, 1]


def test_close_is_idempotent_and_releases_active_environment(fake_stack):
    environment = _setup_environment(fake_stack, num_steps_wait=0)
    environment.reset(environment.tasks()[0], episode_index=0, seed=7)
    simulator = fake_stack.envs.instances[0]

    environment.close()
    environment.close()

    assert simulator.close_calls == 1
    assert environment.render() is None
    with pytest.raises(RuntimeError, match=r'setup\(\) was not called'):
        environment.tasks()
    with pytest.raises(RuntimeError, match=r'setup\(\) was not called'):
        environment.step(np.zeros(LIBERO_ACTION_DIM))


def test_seed_failure_closes_partially_constructed_environment(fake_stack):
    fake_stack.envs.next_seed_error = RuntimeError('seed failed')
    environment = _setup_environment(fake_stack, num_steps_wait=0)

    with pytest.raises(RuntimeError, match='seed failed'):
        environment.reset(environment.tasks()[0], episode_index=0, seed=7)

    simulator = fake_stack.envs.instances[0]
    assert simulator.close_calls == 1
    with pytest.raises(RuntimeError, match=r'reset\(\) must be called'):
        environment.step(np.zeros(LIBERO_ACTION_DIM))


def test_setup_wraps_missing_libero_dependency(monkeypatch):
    monkeypatch.setattr(libero_module, 'configure_mujoco_egl_defaults',
                        lambda: None)

    def missing_import(name):
        raise ModuleNotFoundError(f'No module named {name!r}', name=name)

    monkeypatch.setattr(libero_module.importlib, 'import_module',
                        missing_import)
    environment = LiberoEnvironment(task_suite_name='libero_10')

    with pytest.raises(
            ModuleNotFoundError,
            match=r'bash scripts/install_libero\.sh',
    ) as error:
        environment.setup()

    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert error.value.__cause__.name == 'robosuite'
    environment.close()


def test_python312_collections_compatibility_patch(monkeypatch):
    utils = SimpleNamespace(collections=SimpleNamespace())
    real_import = libero_module.importlib.import_module

    def import_module(name):
        if name == 'libero.libero.envs.utils':
            return utils
        return real_import(name)

    monkeypatch.setattr(libero_module.importlib, 'import_module',
                        import_module)

    libero_module._patch_libero_collections_compatibility()

    assert hasattr(utils.collections, 'Iterable')


@pytest.mark.parametrize(
    'robosuite,error_message',
    [
        (
            SimpleNamespace(
                __version__='1.4.1',
                load_controller_config=lambda: None,
            ),
            'incompatible',
        ),
        (
            SimpleNamespace(__version__='1.5.1'),
            'legacy LIBERO compatibility API',
        ),
    ],
)
def test_runtime_rejects_incompatible_robosuite(monkeypatch, robosuite,
                                                error_message):
    monkeypatch.setattr(libero_module, 'configure_mujoco_egl_defaults',
                        lambda: None)
    modules = {
        'robosuite': robosuite,
        'libero.libero': SimpleNamespace(),
        'libero.libero.benchmark': SimpleNamespace(),
        'libero.libero.envs': SimpleNamespace(),
    }
    monkeypatch.setattr(libero_module.importlib, 'import_module', modules.get)

    with pytest.raises(RuntimeError, match=error_message):
        libero_module._load_libero_runtime()
