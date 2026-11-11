import pytest

from fluxthemis.environments import EnvironmentCaseRejected
from fluxthemis.environments.sim.robodojo import RoboDojoEnvironment
from fluxthemis.structures import TaskSpec


class UnStableError(RuntimeError):
    pass


class _UnstableNativeEnvironment:

    def __init__(self):
        self.reset_calls = []
        self.close_calls = 0

    def reset(self, seed):
        self.reset_calls.append(seed)
        raise UnStableError('unstable scene')

    def close(self):
        self.close_calls += 1


def test_robodojo_translates_unstable_layout_to_case_rejection():
    task = TaskSpec('task')
    native = _UnstableNativeEnvironment()
    environment = object.__new__(RoboDojoEnvironment)
    environment._env = native
    environment._task = task
    environment._task_ids = {task.task_id}
    environment._episode_active = False
    environment._episode_task = None
    environment._episode_index = None
    environment._episode_layout_index = None
    environment._physx_monitor = None
    environment._video_finalized = True

    with pytest.raises(
            EnvironmentCaseRejected, match='RoboDojo layout 17 is unstable'):
        environment.reset(
            task,
            episode_index=0,
            environment_seed=41,
            environment_case_index=17,
        )

    assert native.reset_calls == [[17]]
    assert native.close_calls == 1


class _BatchNativeEnvironment:

    def __init__(self):
        self.reset_calls = []
        self.action_calls = []
        self.close_calls = 0
        self.reward_calls = 0
        self.unstable_envs = set()

    def reset(self, seed):
        self.reset_calls.append(seed)
        self.end_flag = [False] * len(seed)
        self.success = [True] * len(seed)
        self.take_action_cnt = [0] * len(seed)

    def close(self):
        self.close_calls += 1

    def run_reward(self):
        self.reward_calls += 1

    def get_obs_batch(self, env_idx_list=None):
        indices = range(len(
            self.end_flag)) if env_idx_list is None else env_idx_list
        return [{'slot': slot} for slot in indices]

    def take_action_batch(self, actions, env_idx_list):
        self.action_calls.append((actions, env_idx_list))
        for slot in env_idx_list:
            self.take_action_cnt[slot] += 1
        if 0 in env_idx_list:
            self.end_flag[0] = True


@pytest.fixture
def batch_environment(monkeypatch, tmp_path):
    from fluxthemis.environments.base import EnvironmentReset
    from fluxthemis.environments.sim import robodojo

    env = RoboDojoEnvironment(
        task_name='put_bottles_into_dustbin',
        num_envs=3,
        robodojo_root=tmp_path)
    native = _BatchNativeEnvironment()
    env._env = native
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)
    monkeypatch.setattr(env, '_action_to_dict',
                        lambda action: {'action': action})
    requests = [
        EnvironmentReset(env.tasks()[0], i, 17 + i, 10 + i) for i in range(3)
    ]
    return env, native, requests


def test_robodojo_batch_routes_active_slots_and_resets_layouts(
        batch_environment):
    env, native, requests = batch_environment
    observations = env.reset_batch(requests)
    assert native.reset_calls == [[10, 11, 12]]
    assert [info['environment_slot'] for _, info in observations] == [0, 1, 2]
    assert [info['environment_seed']
            for _, info in observations] == [17, 18, 19]
    first = env.step_batch([1, 2, 3])
    assert first[0].terminated and first[0].success
    assert not first[1].terminated
    second = env.step_batch([None, 4, 5])
    assert second[0] is None
    assert second[2].observation == {'slot': 2}
    assert second[2].info['layout_index'] == 12
    assert native.action_calls[-1] == ([{'action': 4}, {'action': 5}], [1, 2])
    with pytest.raises(ValueError, match='reactivate'):
        env.step_batch([1, 4, 5])
    env.reset_batch(requests)
    assert native.close_calls == 1
    assert native.reward_calls == 2
    assert env.step_batch([1, 2, 3])[0].terminated


def test_robodojo_batch_rejects_partial_and_unstable_reset(batch_environment):
    env, native, requests = batch_environment
    with pytest.raises(ValueError, match='full batch'):
        env.reset_batch(requests[:2])
    assert not native.reset_calls
    native.unstable_envs = {1}
    with pytest.raises(RuntimeError, match='unstable slots'):
        env.reset_batch(requests)


def test_robodojo_batch_scores_are_slot_specific(batch_environment):
    from types import SimpleNamespace

    env, native, requests = batch_environment
    native.get_score = lambda: None
    native.reward_manager = SimpleNamespace(get_score=lambda: [100, 25, 75])
    env.reset_batch(requests)
    transitions = env.step_batch([1, 2, 3])
    assert [item.score for item in transitions] == [1.0, 0.25, 0.75]
