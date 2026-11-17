from pathlib import Path
from types import SimpleNamespace

import pytest

from fluxthemis.environments import EnvironmentResetRequest
from fluxthemis.environments.sim import robodojo
from fluxthemis.environments.sim.robodojo import RoboDojoEnvironment


class UnStableError(RuntimeError):
    pass


class _BatchNativeEnvironment:

    def __init__(self):
        self.reset_calls = []
        self.action_calls = []
        self.observation_calls = []
        self.close_calls = 0
        self.reward_calls = 0
        self.unstable_envs = set()
        self.abort_calls = []
        self.video_calls = []

    def reset(self, seed):
        self.reset_calls.append(seed)
        self.end_flag = [value is None for value in seed]
        self.success = [True] * len(seed)
        self.take_action_cnt = [0] * len(seed)

    def close(self):
        self.close_calls += 1

    def run_reward(self):
        self.reward_calls += 1

    def get_obs_batch(self, env_idx_list):
        self.observation_calls.append(list(env_idx_list))
        return [{'slot': slot} for slot in env_idx_list]

    def take_action_batch(self, actions, env_idx_list):
        self.action_calls.append((actions, env_idx_list))
        for slot in env_idx_list:
            self.take_action_cnt[slot] += 1
        if 0 in env_idx_list:
            self.end_flag[0] = True

    def _abort_video_writers(self, env_idx_list):
        self.abort_calls.append(list(env_idx_list))

    def save_video(self, slot, path, tag):
        self.video_calls.append((slot, tag))
        Path(path.replace('.mp4', f'_camera_{tag}.mp4')).write_bytes(b'video')


@pytest.fixture(params=[1, 3])
def batch_environment(monkeypatch, tmp_path, request):
    capacity = request.param
    env = RoboDojoEnvironment(
        task_name='put_bottles_into_dustbin',
        num_envs=capacity,
        robodojo_root=tmp_path)
    native = _BatchNativeEnvironment()
    env._env = native
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)
    monkeypatch.setattr(env, '_action_to_dict',
                        lambda action: {'action': action})
    requests = [
        EnvironmentResetRequest(env.tasks()[0], slot, 17 + slot, 10 + slot)
        for slot in range(capacity)
    ]
    return env, native, requests


def test_routes_active_slots_and_resets_layouts(batch_environment):
    env, native, requests = batch_environment
    capacity = len(requests)
    resets = env.reset_batch(requests)
    assert native.reset_calls == [list(range(10, 10 + capacity))]
    assert [reset.info['environment_slot']
            for reset in resets] == list(range(capacity))
    assert [reset.info['environment_seed']
            for reset in resets] == list(range(17, 17 + capacity))
    assert all(reset.observation['predicates']['environment_success'] is False
               for reset in resets)
    first = env.step_batch(list(range(1, capacity + 1)))
    assert first[0].terminated and first[0].success
    assert all(not result.terminated for result in first[1:])
    second = env.step_batch([None] + list(range(2, capacity + 1)))
    assert second[0] is None
    assert [result.info['layout_index']
            for result in second[1:]] == list(range(11, 10 + capacity))
    assert native.take_action_cnt == [1] + [2] * (capacity - 1)
    assert native.observation_calls[0] == list(range(capacity))
    for actions, slots in native.action_calls:
        assert len(actions) == len(slots)
    with pytest.raises(ValueError, match='reactivate'):
        env.step_batch([1] + [None] * (capacity - 1))
    env.reset_batch(requests)
    assert native.close_calls == 1
    assert native.reward_calls == 2
    assert env.step_batch(list(range(1, capacity + 1)))[0].terminated


def test_rejects_only_unstable_reset_slot(batch_environment):
    env, native, requests = batch_environment
    with pytest.raises(ValueError, match='one entry per slot'):
        env.reset_batch(requests[:-1])
    assert not native.reset_calls
    rejected_slot = len(requests) - 1
    native.unstable_envs = {rejected_slot}
    resets = env.reset_batch(requests)
    assert [reset.rejected
            for reset in resets] == [False] * rejected_slot + [True]
    assert resets[rejected_slot].info[
        'environment_case_index'] == 10 + rejected_slot
    assert native.abort_calls == [[rejected_slot]]
    transitions = env.step_batch([1] * rejected_slot + [None])
    assert transitions[rejected_slot] is None
    assert all(result is not None for result in transitions[:rejected_slot])


def test_unidentified_native_reset_failure_returns_slot_rejections(
        batch_environment, monkeypatch):
    env, native, requests = batch_environment

    def unstable(seed):
        native.reset_calls.append(seed)
        raise UnStableError('unstable scene')

    monkeypatch.setattr(native, 'reset', unstable)
    resets = env.reset_batch(requests)
    assert all(reset.rejected for reset in resets)
    assert len(resets) == len(requests)
    assert not native.observation_calls


def test_scores_are_slot_specific(batch_environment):
    env, native, requests = batch_environment
    scores = [100, 25, 75][:len(requests)]
    native.get_score = lambda: None
    native.reward_manager = SimpleNamespace(get_score=lambda: scores)
    env.reset_batch(requests)
    transitions = env.step_batch([1] * len(requests))
    assert [item.score
            for item in transitions] == [score / 100 for score in scores]


def test_tail_batch_masks_unused_native_lanes(batch_environment):
    env, native, requests = batch_environment
    inactive = [None] * (len(requests) - 1)
    native.unstable_envs = set(range(1, len(requests)))
    resets = env.reset_batch([requests[0], *inactive])
    assert not resets[0].rejected
    assert resets[1:] == inactive
    assert native.reset_calls == [[10, *inactive]]
    transitions = env.step_batch([1, *inactive])
    assert transitions[1:] == inactive
    assert native.action_calls[-1][1] == [0]


def test_all_unused_reset_slots_do_not_start_native_episodes(
        batch_environment):
    env, native, requests = batch_environment
    inactive = [None] * len(requests)
    assert env.reset_batch(inactive) == inactive
    assert not native.reset_calls


def test_runtime_invalid_layout_freezes_only_its_lane(batch_environment,
                                                      monkeypatch):
    env, native, requests = batch_environment
    capacity = len(requests)
    rejected_slot = capacity - 1
    env.reset_batch(requests)
    take_action = native.take_action_batch

    def invalidate(actions, env_idx_list):
        take_action(actions, env_idx_list)
        native.unstable_envs.add(rejected_slot)

    monkeypatch.setattr(native, 'take_action_batch', invalidate)
    transitions = env.step_batch([1] * capacity)
    assert transitions[rejected_slot].case_rejected
    assert transitions[rejected_slot].success is None
    assert all(not result.case_rejected
               for result in transitions[:rejected_slot])
    assert native.abort_calls == [[rejected_slot]]
    actions = [None] * capacity
    for slot in range(1, rejected_slot):
        actions[slot] = 2
    transitions = env.step_batch(actions)
    assert transitions[0] is None and transitions[rejected_slot] is None
    assert all(transitions[slot] is not None
               for slot in range(1, rejected_slot))
    assert native.take_action_cnt == [
        1 + int(1 <= slot < rejected_slot) for slot in range(capacity)
    ]


def test_partial_native_reset_failure_keeps_healthy_slots(
        batch_environment, monkeypatch):
    env, native, requests = batch_environment
    reset = native.reset
    rejected_slot = len(requests) - 1

    def unstable(seed):
        reset(seed)
        native.end_flag[rejected_slot] = True
        raise UnStableError('partial reset')

    monkeypatch.setattr(native, 'reset', unstable)
    resets = env.reset_batch(requests)
    assert [reset.rejected
            for reset in resets] == [False] * rejected_slot + [True]
    monkeypatch.setattr(native, 'reset', reset)
    env.reset_batch(requests)
    assert native.close_calls == 1


def test_rejection_during_initial_observation_is_slot_specific(
        batch_environment, monkeypatch):
    env, native, requests = batch_environment
    get_observations = native.get_obs_batch
    rejected_slot = len(requests) - 1

    def unstable(env_idx_list):
        observations = get_observations(env_idx_list)
        native.unstable_envs.add(rejected_slot)
        return observations

    monkeypatch.setattr(native, 'get_obs_batch', unstable)
    resets = env.reset_batch(requests)
    assert [reset.rejected
            for reset in resets] == [False] * rejected_slot + [True]


def test_policy_stop_freezes_its_slot_without_changing_native_success(
        batch_environment):
    env, native, requests = batch_environment
    slot = len(requests) - 1
    env.reset_batch(requests)
    before = (list(native.end_flag), list(native.success))
    assert env.finalize_episode(
        slot, success=False, termination_reason='policy_completed') == {}
    assert (native.end_flag, native.success) == before
    transitions = env.step_batch([1] * slot + [None])
    assert transitions[slot] is None
    assert native.take_action_cnt == [1] * slot + [0]
    with pytest.raises(ValueError, match='reactivate'):
        env.step_batch([None] * slot + [1])


@pytest.mark.parametrize('capacity', [1, 3])
def test_completion_diagnostics_use_each_native_slot(monkeypatch, tmp_path,
                                                     capacity):
    env = RoboDojoEnvironment(
        task_name='stack_blocks',
        num_envs=capacity,
        completion_diagnostics=True,
        robodojo_root=tmp_path)
    native = _BatchNativeEnvironment()
    native._stacked_xy_thresholds = lambda: 0.01
    values = {'stacked': [1, 0, 1], 'home': [0, 1, 1]}
    native.reward_manager = SimpleNamespace(
        is_stacked=lambda **kwargs: 'stacked',
        all_robot_back_to_origin=lambda: 'home',
        call_func_parser=lambda check, slot: values[check][slot])
    env._env = native
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)
    monkeypatch.setattr(env, '_action_to_dict', lambda action: action)
    requests = [
        EnvironmentResetRequest(env.tasks()[0], slot, slot, slot)
        for slot in range(capacity)
    ]
    resets = env.reset_batch(requests)
    for slot, reset in enumerate(resets):
        assert reset.observation['predicates']['blocks_stacked'] == bool(
            values['stacked'][slot])
        assert reset.observation['predicates']['arms_at_origin'] == bool(
            values['home'][slot])
        assert reset.info['completion_predicates'] == reset.observation[
            'predicates']
    transitions = env.step_batch([1] * capacity)
    assert [
        result.observation['predicates']['environment_success']
        for result in transitions
    ] == [True] + [False] * (
        capacity - 1)


def test_native_video_finalizes_and_aborts_each_slot(batch_environment,
                                                     tmp_path):
    env, native, requests = batch_environment
    env.save_videos = True
    env._video_dir = tmp_path / 'videos'
    env._video_dir.mkdir()
    native.save_dir = tmp_path / 'native'
    Path(native.save_dir).mkdir()
    env.reset_batch(requests)
    for slot, request in enumerate(requests):
        info = env.finalize_episode(
            slot, success=False, termination_reason='policy_completed')
        assert native.video_calls[-1] == (slot, 'agent_done')
        path, = info['video_paths']
        assert Path(path).parent == env._video_dir
        assert f'episode_{request.episode_index:07d}' in Path(path).name
        assert Path(path).read_bytes() == b'video'
        assert env.finalize_episode(
            slot, success=False, termination_reason='policy_completed') == {}
    env.reset_batch(requests)
    rejected_slot = len(requests) - 1
    assert env.finalize_episode(
        rejected_slot, success=False,
        termination_reason='case_rejected') == {}
    assert native.abort_calls == [[rejected_slot]]
    env.close()
    assert [slot for slot, _ in native.video_calls] == (
        list(range(len(requests))) + list(range(rejected_slot)))
