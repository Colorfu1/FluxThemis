import numpy as np
import pytest

from fluxthemis.environments.real import BaseRealRobotEnvironment
from fluxthemis.structures import StepResult, TaskSpec


class _RobotEnvironment(BaseRealRobotEnvironment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connected = False
        self.stopped = False

    def connect(self):
        self.connected = True

    def tasks(self):
        return [TaskSpec('robot')]

    def reset(self, task, episode_index, seed):
        return {}, {}

    def execute_action(self, action):
        return StepResult(observation={}, info={})

    def on_emergency_stop(self):
        self.stopped = True

    def disconnect(self):
        self.connected = False


class _PartiallyConnectedRobot(_RobotEnvironment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disconnect_called = False

    def connect(self):
        self.connected = True
        raise RuntimeError('connect failed')

    def disconnect(self):
        self.disconnect_called = True
        self.connected = False


def test_real_robot_actuation_is_disabled_by_default():
    environment = _RobotEnvironment()
    with pytest.raises(RuntimeError, match='actuation is disabled'):
        environment.setup()


def test_real_robot_requires_explicit_action_bounds():
    environment = _RobotEnvironment(allow_actuation=True)
    with pytest.raises(RuntimeError, match='requires explicit'):
        environment.setup()


def test_real_robot_rejects_unsafe_bounds():
    with pytest.raises(ValueError, match='cannot exceed'):
        _RobotEnvironment(
            allow_actuation=True,
            action_low=[1.0],
            action_high=[-1.0],
        )
    with pytest.raises(ValueError, match='finite'):
        _RobotEnvironment(
            allow_actuation=True,
            action_low=[-np.inf],
            action_high=[1.0],
        )


def test_real_robot_validates_every_action():
    environment = _RobotEnvironment(
        allow_actuation=True,
        action_low=[-1.0, -2.0],
        action_high=[1.0, 2.0],
    )
    environment.setup()
    assert environment.connected
    environment.step([0.5, -1.0])
    with pytest.raises(ValueError, match='exceeds'):
        environment.step([1.5, 0.0])
    environment.emergency_stop()
    assert environment.stopped
    with pytest.raises(RuntimeError, match='emergency-stopped'):
        environment.step([0.0, 0.0])
    environment.close()
    assert not environment.connected


def test_failed_connect_releases_partially_allocated_resources():
    environment = _PartiallyConnectedRobot(
        allow_actuation=True,
        action_low=[-1.0],
        action_high=[1.0],
    )
    with pytest.raises(RuntimeError, match='connect failed'):
        environment.setup()
    assert environment.disconnect_called
    assert not environment.connected
