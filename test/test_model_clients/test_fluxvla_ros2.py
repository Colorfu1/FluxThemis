import json
import sys
import types
import uuid

import numpy as np
import pytest

from fluxthemis.engines import MODEL_CLIENTS
from fluxthemis.model_clients.fluxvla_ros2 import FluxVLAROS2ModelClient
from fluxthemis.structures import TaskSpec


class _Header:

    def __init__(self):
        self.stamp = None


class _PredictActionRequest:

    def __init__(self):
        self.header = _Header()
        self.request_id = ''
        self.episode_id = ''
        self.reset = False
        self.seed = 0
        self.prompt = ''
        self.image_names = []
        self.images = []
        self.state_names = []
        self.state_sizes = []
        self.state_values = []
        self.unnorm_key = ''


class _PredictAction:
    Request = _PredictActionRequest


class _ReportEvaluationRequest:

    def __init__(self):
        self.header = _Header()
        self.request_id = ''
        self.run_id = ''
        self.sequence = 0
        self.event_type = ''
        self.payload_json = ''


class _ReportEvaluation:
    Request = _ReportEvaluationRequest


class _Future:

    def __init__(self, response=None, exception=None, done=True):
        self._response = response
        self._exception = exception
        self._done = done
        self.cancelled = False

    def done(self):
        return self._done

    def result(self):
        return self._response

    def exception(self):
        return self._exception

    def cancel(self):
        self.cancelled = True


class _Client:

    def __init__(self, runtime, service_name):
        self.runtime = runtime
        self.service_name = service_name

    def wait_for_service(self, timeout_sec):
        self.runtime.waits.append(timeout_sec)
        return self.runtime.service_availability.get(
            self.service_name, self.runtime.service_available)

    def call_async(self, request):
        self.runtime.requests.append(request)
        self.runtime.requests_by_service.setdefault(self.service_name,
                                                    []).append(request)
        future_factory = self.runtime.future_factories.get(
            self.service_name, self.runtime.future_factory)
        future = future_factory(request)
        self.runtime.futures.append(future)
        return future

    def remove_pending_request(self, future):
        self.runtime.removed_futures.append(future)


class _ClockNow:

    @staticmethod
    def to_msg():
        return 'fake-ros2-time'


class _Clock:

    @staticmethod
    def now():
        return _ClockNow()


class _Node:

    def __init__(self, runtime):
        self.runtime = runtime
        self.client = None
        self.clients = []
        self.destroyed = False

    def create_client(self, service_type, service_name):
        self.runtime.create_client_calls.append((service_type, service_name))
        client = _Client(self.runtime, service_name)
        self.clients.append(client)
        if self.client is None:
            self.client = client
        return client

    def destroy_client(self, client):
        self.runtime.destroyed_clients.append(client)

    def destroy_node(self):
        self.destroyed = True

    @staticmethod
    def get_clock():
        return _Clock()


class _FakeROS2Runtime:

    def __init__(self):
        self.initialized = False
        self.init_calls = []
        self.shutdown_calls = 0
        self.node_names = []
        self.create_client_calls = []
        self.destroyed_clients = []
        self.waits = []
        self.requests = []
        self.requests_by_service = {}
        self.futures = []
        self.removed_futures = []
        self.spin_calls = []
        self.service_available = True
        self.service_availability = {}
        self.node = None
        self.future_factory = self.valid_future
        self.future_factories = {}

    def init(self, args):
        self.init_calls.append(args)
        self.initialized = True

    def try_shutdown(self):
        self.shutdown_calls += 1
        self.initialized = False

    def create_node(self, name):
        self.node_names.append(name)
        self.node = _Node(self)
        return self.node

    def spin_until_future_complete(self, node, future, timeout_sec):
        self.spin_calls.append((node, future, timeout_sec))

    @staticmethod
    def valid_response(request):
        return types.SimpleNamespace(
            header=_Header(),
            request_id=request.request_id,
            ok=True,
            error='',
            actions=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            action_horizon=2,
            action_dim=3,
            denormalized=True,
            inference_time_s=0.02,
        )

    def valid_future(self, request):
        return _Future(response=self.valid_response(request))


@pytest.fixture
def fake_ros2(monkeypatch):
    runtime = _FakeROS2Runtime()

    rclpy = types.ModuleType('rclpy')
    rclpy.ok = lambda: runtime.initialized
    rclpy.init = runtime.init
    rclpy.try_shutdown = runtime.try_shutdown
    rclpy.create_node = runtime.create_node
    rclpy.spin_until_future_complete = runtime.spin_until_future_complete

    class CvBridge:

        @staticmethod
        def cv2_to_imgmsg(image, encoding):
            return types.SimpleNamespace(
                image=np.asarray(image).copy(),
                encoding=encoding,
            )

    cv_bridge = types.ModuleType('cv_bridge')
    cv_bridge.CvBridge = CvBridge

    service_module = types.ModuleType('fluxthemis_msgs.srv')
    service_module.PredictAction = _PredictAction
    service_module.ReportEvaluation = _ReportEvaluation
    message_package = types.ModuleType('fluxthemis_msgs')
    message_package.__path__ = []
    message_package.srv = service_module

    monkeypatch.setitem(sys.modules, 'rclpy', rclpy)
    monkeypatch.setitem(sys.modules, 'cv_bridge', cv_bridge)
    monkeypatch.setitem(sys.modules, 'fluxthemis_msgs', message_package)
    monkeypatch.setitem(sys.modules, 'fluxthemis_msgs.srv', service_module)
    return runtime


def _transport(**overrides):
    transport = {
        'service_name': '/policy/predict_action',
        'timeout_s': 2.5,
        'request_timeout_s': 1.25,
        'image_keys': ['front'],
        'state_keys': ['proprio'],
        'unnorm_key': 'robot',
        'image_encoding': 'rgb8',
    }
    transport.update(overrides)
    return transport


def _observation():
    return {
        'front': np.zeros((4, 5, 3), dtype=np.uint8),
        'proprio': np.asarray([0.25, -1.5, 3.0], dtype=np.float32),
    }


def _ready_client(fake_ros2):
    client = FluxVLAROS2ModelClient(transport=_transport())
    client.setup()
    client.reset(
        TaskSpec(task_id='pick', description='pick up the block'),
        episode_index=3,
        seed=17,
    )
    return client


def test_ros2_client_is_registered_without_importing_ros2():
    assert MODEL_CLIENTS.get('FluxVLAROS2ModelClient') is \
        FluxVLAROS2ModelClient


@pytest.mark.parametrize(
    'value,exception',
    [
        (0.0, ValueError),
        (float('inf'), ValueError),
        (True, TypeError),
        ('1', TypeError),
    ],
)
def test_request_timeout_must_be_finite_and_positive(value, exception):
    transport = _transport()
    transport['request_timeout_s'] = value

    with pytest.raises(exception, match='request_timeout_s'):
        FluxVLAROS2ModelClient(transport=transport)


def test_ros1_persistent_option_is_not_forwarded_as_ros2_transport():
    transport = _transport()
    transport['persistent'] = False

    client = FluxVLAROS2ModelClient(transport=transport)

    assert 'persistent' not in client.transport


def test_setup_predict_and_owned_context_cleanup(fake_ros2):
    client = _ready_client(fake_ros2)

    chunk = client.predict_action(_observation(), {
        'prompt': 'move the block',
        'unnorm_key': 'environment_units',
    })

    assert fake_ros2.init_calls == [None]
    assert fake_ros2.node_names == ['fluxthemis_eval']
    assert fake_ros2.create_client_calls == [(_PredictAction,
                                              '/policy/predict_action')]
    assert fake_ros2.waits == [2.5]
    assert len(fake_ros2.requests) == 1
    request = fake_ros2.requests[0]
    assert request.header.stamp == 'fake-ros2-time'
    assert request.episode_id == 'pick:3'
    assert request.reset is True
    assert request.seed == 17
    assert request.prompt == 'move the block'
    assert request.image_names == ['front']
    assert request.images[0].encoding == 'rgb8'
    assert request.state_names == ['proprio']
    assert request.state_sizes == [3]
    assert request.unnorm_key == 'environment_units'
    assert fake_ros2.spin_calls[0][2] == 1.25
    np.testing.assert_array_equal(
        chunk.actions,
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )
    assert chunk.metadata['denormalized'] is True

    node = fake_ros2.node
    service = node.client
    client.close()
    client.close()
    assert fake_ros2.destroyed_clients == [service]
    assert node.destroyed is True
    assert fake_ros2.shutdown_calls == 1


def test_ros2_reporting_sends_acknowledged_uuid_session_and_sequence(
        fake_ros2):

    def report_future(request):
        return _Future(
            response=types.SimpleNamespace(
                header=_Header(),
                request_id=request.request_id,
                accepted=True,
                error='',
                run_dir='/srv/fluxvla/evaluations/ros2-run',
            ))

    fake_ros2.future_factories['/policy/report_evaluation'] = report_future
    client = FluxVLAROS2ModelClient(
        transport=_transport(report_service_name='/policy/report_evaluation'))

    client.setup()
    client.report_evaluation('run_start', {
        'run_name': 'acceptance',
        'full_suite': True,
    })
    client.report_evaluation('run_end', {'status': 'finished'})

    assert fake_ros2.create_client_calls == [
        (_PredictAction, '/policy/predict_action'),
        (_ReportEvaluation, '/policy/report_evaluation'),
    ]
    assert fake_ros2.waits == [2.5, 2.5]
    requests = fake_ros2.requests_by_service['/policy/report_evaluation']
    first, second = requests
    uuid.UUID(first.request_id)
    uuid.UUID(first.run_id)
    assert first.header.stamp == 'fake-ros2-time'
    assert first.run_id == second.run_id == client.evaluation_report_run_id
    assert [first.sequence, second.sequence] == [1, 2]
    assert json.loads(first.payload_json) == {
        'run_name': 'acceptance',
        'full_suite': True,
    }
    assert client.evaluation_report_run_dir == (
        '/srv/fluxvla/evaluations/ros2-run')

    node = fake_ros2.node
    client.close()
    assert fake_ros2.destroyed_clients == node.clients


def test_ros2_reporting_retries_async_transport_with_stable_request(
        fake_ros2, monkeypatch):
    attempts = 0

    def report_future(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return _Future(exception=RuntimeError('temporary failure'))
        return _Future(
            response=types.SimpleNamespace(
                request_id=request.request_id,
                accepted=True,
                error='',
                run_dir='/srv/fluxvla/evaluations/ros2-retry',
            ))

    fake_ros2.future_factories['/policy/report_evaluation'] = report_future
    monkeypatch.setattr('fluxthemis.model_clients.fluxvla_ros.time.sleep',
                        lambda _: None)
    client = FluxVLAROS2ModelClient(
        transport=_transport(report_service_name='/policy/report_evaluation'))
    client.setup()

    client.report_evaluation('run_start', {'run_name': 'retry'})

    requests = fake_ros2.requests_by_service['/policy/report_evaluation']
    assert len(requests) == 3
    assert len({request.request_id for request in requests}) == 1
    assert {request.sequence for request in requests} == {1}
    assert len(fake_ros2.spin_calls) == 3


def test_setup_reuses_existing_context_without_shutting_it_down(fake_ros2):
    fake_ros2.initialized = True
    client = FluxVLAROS2ModelClient(transport=_transport())

    client.setup()
    client.close()

    assert fake_ros2.init_calls == []
    assert fake_ros2.shutdown_calls == 0


def test_setup_timeout_cleans_up_owned_node_and_context(fake_ros2):
    fake_ros2.service_available = False
    client = FluxVLAROS2ModelClient(transport=_transport())

    with pytest.raises(TimeoutError, match='was not available'):
        client.setup()

    assert fake_ros2.node.destroyed is True
    assert fake_ros2.destroyed_clients == [fake_ros2.node.client]
    assert fake_ros2.shutdown_calls == 1


def test_report_service_timeout_cleans_up_both_clients(fake_ros2):
    fake_ros2.service_availability['/policy/report_evaluation'] = False
    client = FluxVLAROS2ModelClient(
        transport=_transport(report_service_name='/policy/report_evaluation'))

    with pytest.raises(TimeoutError, match='report_evaluation'):
        client.setup()

    assert fake_ros2.destroyed_clients == fake_ros2.node.clients
    assert fake_ros2.node.destroyed is True
    assert fake_ros2.shutdown_calls == 1


def test_request_timeout_cancels_future_and_keeps_episode_reset(fake_ros2):
    fake_ros2.future_factory = lambda request: _Future(done=False)
    client = _ready_client(fake_ros2)

    with pytest.raises(TimeoutError, match='did not respond'):
        client.predict_action(_observation(), {'prompt': 'task'})
    with pytest.raises(TimeoutError, match='did not respond'):
        client.predict_action(_observation(), {'prompt': 'task'})

    assert [future.cancelled for future in fake_ros2.futures] == [True, True]
    assert fake_ros2.removed_futures == fake_ros2.futures
    assert [request.reset for request in fake_ros2.requests] == [True, True]


def test_request_propagates_async_service_error(fake_ros2):
    service_error = RuntimeError('transport failed')
    fake_ros2.future_factory = lambda request: _Future(exception=service_error)
    client = _ready_client(fake_ros2)

    with pytest.raises(RuntimeError, match='call failed') as exc_info:
        client.predict_action(_observation(), {'prompt': 'task'})

    assert exc_info.value.__cause__ is service_error


def test_predict_requires_setup_and_reset(fake_ros2):
    client = FluxVLAROS2ModelClient(transport=_transport())
    client.reset(TaskSpec(task_id='task', description='task'), 0, 1)
    with pytest.raises(RuntimeError, match=r'setup\(\)'):
        client.predict_action(_observation(), {'prompt': 'task'})

    client = FluxVLAROS2ModelClient(transport=_transport())
    client.setup()
    with pytest.raises(RuntimeError, match=r'reset\(\)'):
        client.predict_action(_observation(), {'prompt': 'task'})
