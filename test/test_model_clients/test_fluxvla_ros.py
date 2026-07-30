import json
import sys
import types
import uuid

import numpy as np
import pytest

from fluxthemis.engines import MODEL_CLIENTS
from fluxthemis.model_clients.fluxvla_ros import (FluxVLAROSModelClient,
                                                  _RawImageBridge)
from fluxthemis.structures import TaskSpec


class _Header:

    def __init__(self):
        self.stamp = None


class _PredictAction:
    pass


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


class _ReportEvaluation:
    pass


class _ReportEvaluationRequest:

    def __init__(self):
        self.header = _Header()
        self.request_id = ''
        self.run_id = ''
        self.sequence = 0
        self.event_type = ''
        self.payload_json = ''


class _Image:

    def __init__(self):
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.encoding = ''
        self.is_bigendian = 0
        self.step = 0
        self.data = b''


class _FakeServiceProxy:

    def __init__(self, runtime, service_name):
        self.runtime = runtime
        self.service_name = service_name
        self.closed = False

    def __call__(self, request):
        if self.service_name == '/policy/report_evaluation':
            self.runtime.report_requests.append(request)
            return self.runtime.report_responder(request)
        self.runtime.requests.append(request)
        return self.runtime.responder(request)

    def close(self):
        self.closed = True


class _FakeROSRuntime:

    def __init__(self):
        self.waits = []
        self.proxy_calls = []
        self.requests = []
        self.init_calls = []
        self.initialized = False
        self.proxy = None
        self.proxies = {}
        self.unavailable_services = set()
        self.responder = self.valid_response
        self.report_requests = []
        self.report_responder = self.valid_report_response

    def wait_for_service(self, service_name, timeout):
        self.waits.append((service_name, timeout))
        if service_name in self.unavailable_services:
            raise TimeoutError(f'{service_name} unavailable')

    def init_node(self, name, anonymous, disable_signals):
        self.init_calls.append((name, anonymous, disable_signals))
        self.initialized = True

    def service_proxy(self, service_name, service_type, persistent):
        self.proxy_calls.append((service_name, service_type, persistent))
        proxy = _FakeServiceProxy(self, service_name)
        self.proxies[service_name] = proxy
        if service_name == '/policy/predict_action':
            self.proxy = proxy
        return proxy

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
            inference_time_s=0.0125,
        )

    @staticmethod
    def valid_report_response(request):
        return types.SimpleNamespace(
            header=_Header(),
            request_id=request.request_id,
            accepted=True,
            error='',
            run_dir='/srv/fluxvla/evaluations/run-1',
        )


@pytest.fixture
def fake_ros(monkeypatch):
    runtime = _FakeROSRuntime()

    rospy = types.ModuleType('rospy')
    rospy.Time = types.SimpleNamespace(now=lambda: 'fake-ros-time')
    rospy.core = types.SimpleNamespace(
        is_initialized=lambda: runtime.initialized)
    rospy.init_node = runtime.init_node
    rospy.wait_for_service = runtime.wait_for_service
    rospy.ServiceProxy = runtime.service_proxy

    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msgs.__path__ = []
    sensor_msgs_msg = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg.Image = _Image
    sensor_msgs.msg = sensor_msgs_msg

    service_module = types.ModuleType('fluxthemis_msgs.srv')
    service_module.PredictAction = _PredictAction
    service_module.PredictActionRequest = _PredictActionRequest
    service_module.ReportEvaluation = _ReportEvaluation
    service_module.ReportEvaluationRequest = _ReportEvaluationRequest
    message_package = types.ModuleType('fluxthemis_msgs')
    message_package.__path__ = []
    message_package.srv = service_module

    monkeypatch.setitem(sys.modules, 'rospy', rospy)
    monkeypatch.setitem(sys.modules, 'sensor_msgs', sensor_msgs)
    monkeypatch.setitem(sys.modules, 'sensor_msgs.msg', sensor_msgs_msg)
    monkeypatch.setitem(sys.modules, 'fluxthemis_msgs', message_package)
    monkeypatch.setitem(sys.modules, 'fluxthemis_msgs.srv', service_module)
    return runtime


def _transport(**overrides):
    transport = {
        'service_name': '/policy/predict_action',
        'timeout_s': 2.5,
        'image_keys': ['front', 'wrist'],
        'state_keys': ['proprio', 'gripper'],
        'unnorm_key': 'default_robot',
        'image_encoding': 'rgb8',
        'persistent': True,
    }
    transport.update(overrides)
    return transport


def _observation():
    return {
        'front': np.zeros((4, 5, 3), dtype=np.uint8),
        'wrist': np.full((2, 3, 3), 7, dtype=np.uint8),
        'proprio': np.asarray([0.25, -1.5, 3.0], dtype=np.float64),
        'gripper': np.asarray([0.04], dtype=np.float32),
    }


def _ready_client(fake_ros):
    client = FluxVLAROSModelClient(transport=_transport())
    client.setup()
    client.reset(
        TaskSpec(task_id='pick', description='pick up the block'),
        episode_index=3,
        seed=17,
    )
    return client


def test_ros_client_is_registered_without_importing_ros():
    assert MODEL_CLIENTS.get('FluxVLAROSModelClient') is \
        FluxVLAROSModelClient


def test_setup_reuses_an_existing_ros_node(fake_ros):
    fake_ros.initialized = True
    client = FluxVLAROSModelClient(transport=_transport())

    client.setup()

    assert fake_ros.init_calls == []


def test_setup_predict_and_close_with_fake_ros_modules(fake_ros):
    client = _ready_client(fake_ros)
    client.setup()

    first_chunk = client.predict_action(
        _observation(), {
            'prompt': 'move the block into the bowl',
            'unnorm_key': 'robot_environment_units',
        })
    second_chunk = client.predict_action(
        _observation(), {'task_description': 'continue the task'})

    assert fake_ros.waits == [('/policy/predict_action', 2.5)]
    assert fake_ros.init_calls == [('fluxthemis_eval', True, True)]
    assert fake_ros.proxy_calls == [('/policy/predict_action', _PredictAction,
                                     True)]
    assert len(fake_ros.requests) == 2
    first_request, second_request = fake_ros.requests
    assert first_request.header.stamp == 'fake-ros-time'
    assert first_request.request_id != second_request.request_id
    assert first_request.episode_id == 'pick:3'
    assert first_request.reset is True
    assert second_request.reset is False
    assert first_request.seed == 17
    assert first_request.prompt == 'move the block into the bowl'
    assert first_request.image_names == ['front', 'wrist']
    assert [message.encoding
            for message in first_request.images] == ['rgb8', 'rgb8']
    np.testing.assert_array_equal(
        np.frombuffer(first_request.images[1].data, dtype=np.uint8).reshape(
            first_request.images[1].height,
            first_request.images[1].width,
            3,
        ),
        _observation()['wrist'],
    )
    assert first_request.images[1].step == 9
    assert first_request.images[1].is_bigendian == 0
    np.testing.assert_allclose(
        first_request.state_values,
        np.asarray([0.25, -1.5, 3.0, 0.04], dtype=np.float32),
    )
    assert first_request.state_names == ['proprio', 'gripper']
    assert first_request.state_sizes == [3, 1]
    assert first_request.unnorm_key == 'robot_environment_units'
    assert second_request.unnorm_key == 'default_robot'

    assert first_chunk.actions.shape == (2, 3)
    assert first_chunk.actions.dtype == np.float32
    np.testing.assert_array_equal(
        first_chunk.actions,
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )
    assert first_chunk.metadata['request_id'] == first_request.request_id
    assert first_chunk.metadata['episode_id'] == 'pick:3'
    assert first_chunk.metadata['reset'] is True
    assert first_chunk.metadata['denormalized'] is True
    assert first_chunk.metadata['profile']['server_infer_s'] == 0.0125
    assert second_chunk.metadata['reset'] is False

    proxy = fake_ros.proxy
    client.close()
    client.close()
    assert proxy.closed is True


def test_explicit_episode_id_is_used_for_parallel_inference(fake_ros):
    client = _ready_client(fake_ros)

    client.set_episode_id('report-run-id:4')
    chunk = client.predict_action(_observation(), {'prompt': 'pick'})

    assert client.supports_explicit_episode_id is True
    assert fake_ros.requests[0].episode_id == 'report-run-id:4'
    assert fake_ros.requests[0].reset is True
    assert chunk.metadata['episode_id'] == 'report-run-id:4'


def test_reporting_setup_sends_acknowledged_uuid_session_and_sequence(
        fake_ros):
    client = FluxVLAROSModelClient(
        transport=_transport(report_service_name='/policy/report_evaluation'))

    client.setup()
    first_run_dir = client.report_evaluation('run_start', {
        'run_name': 'acceptance',
        'seed': 7,
    })
    second_run_dir = client.report_evaluation('episode_start', {
        'task_id': 'pick',
        'task_index': 0,
    })

    assert client.evaluation_reporting_enabled is True
    assert first_run_dir == second_run_dir == (
        '/srv/fluxvla/evaluations/run-1')
    assert client.evaluation_report_run_dir == first_run_dir
    assert fake_ros.waits == [
        ('/policy/predict_action', 2.5),
        ('/policy/report_evaluation', 2.5),
    ]
    assert fake_ros.proxy_calls == [
        ('/policy/predict_action', _PredictAction, True),
        ('/policy/report_evaluation', _ReportEvaluation, True),
    ]
    first, second = fake_ros.report_requests
    uuid.UUID(first.request_id)
    uuid.UUID(first.run_id)
    assert first.header.stamp == 'fake-ros-time'
    assert first.run_id == second.run_id == client.evaluation_report_run_id
    assert first.request_id != second.request_id
    assert [first.sequence, second.sequence] == [1, 2]
    assert [first.event_type,
            second.event_type] == ['run_start', 'episode_start']
    assert json.loads(first.payload_json) == {
        'run_name': 'acceptance',
        'seed': 7,
    }

    client.close()
    assert all(proxy.closed for proxy in fake_ros.proxies.values())


def test_reporting_setup_failure_closes_action_proxy(fake_ros):
    fake_ros.unavailable_services.add('/policy/report_evaluation')
    client = FluxVLAROSModelClient(
        transport=_transport(report_service_name='/policy/report_evaluation'))

    with pytest.raises(TimeoutError, match='report_evaluation'):
        client.setup()

    assert fake_ros.proxy.closed is True


def test_reporting_retries_transport_with_stable_id_and_sequence(
        fake_ros, monkeypatch):
    attempts = 0
    sleeps = []

    def flaky_report(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError('connection reset')
        return fake_ros.valid_report_response(request)

    fake_ros.report_responder = flaky_report
    monkeypatch.setattr('fluxthemis.model_clients.fluxvla_ros.time.sleep',
                        sleeps.append)
    client = FluxVLAROSModelClient(
        transport=_transport(report_service_name='/policy/report_evaluation'))
    client.setup()

    client.report_evaluation('run_start', {'run_name': 'retry'})
    client.report_evaluation('run_end', {'status': 'finished'})

    retried = fake_ros.report_requests[:3]
    assert len({request.request_id for request in retried}) == 1
    assert len({request.run_id for request in retried}) == 1
    assert {request.sequence for request in retried} == {1}
    assert fake_ros.report_requests[-1].sequence == 2
    assert sleeps == [0.1, 0.2]


@pytest.mark.parametrize(
    'mutate_response,message',
    [
        (lambda response: setattr(response, 'request_id', 'stale'),
         'request_id mismatch'),
        (lambda response:
         (setattr(response, 'accepted', False),
          setattr(response, 'error', 'disk full')), 'disk full'),
        (lambda response: setattr(response, 'run_dir', ''), 'empty run_dir'),
    ],
)
def test_reporting_validates_acknowledgements(fake_ros, mutate_response,
                                              message):

    def responder(request):
        response = fake_ros.valid_report_response(request)
        mutate_response(response)
        return response

    fake_ros.report_responder = responder
    client = FluxVLAROSModelClient(
        transport=_transport(report_service_name='/policy/report_evaluation'))
    client.setup()

    with pytest.raises(RuntimeError, match=message):
        client.report_evaluation('run_start', {'run_name': 'invalid-ack'})


def test_reporting_rejects_changed_authoritative_run_dir(fake_ros):
    client = FluxVLAROSModelClient(
        transport=_transport(report_service_name='/policy/report_evaluation'))
    client.setup()
    client.report_evaluation('run_start', {'run_name': 'run-dir'})

    def changed_run_dir(request):
        response = fake_ros.valid_report_response(request)
        response.run_dir = '/srv/fluxvla/evaluations/other-run'
        return response

    fake_ros.report_responder = changed_run_dir
    with pytest.raises(RuntimeError, match='changed the authoritative'):
        client.report_evaluation('run_end', {'status': 'finished'})


@pytest.mark.parametrize(
    'transport,exception,message',
    [
        (None, TypeError, 'mapping'),
        ({}, ValueError, 'image_keys is required'),
        ({
            'image_keys': ['front']
        }, ValueError, 'state_keys is required'),
        ({
            'image_keys': [],
            'state_keys': ['state']
        }, ValueError, 'cannot be empty'),
        ({
            'image_keys': ['front', 'front'],
            'state_keys': ['state']
        }, ValueError, 'duplicates'),
        ({
            'image_keys': ['front'],
            'state_keys': ['state', 'state']
        }, ValueError, 'duplicates'),
        ({
            'image_keys': ['front'],
            'state_keys': ['state'],
            'unknown': True
        }, ValueError, 'Unknown'),
        ({
            'image_keys': ['front'],
            'state_keys': ['state'],
            'timeout_s': 0
        }, ValueError, 'positive'),
        ({
            'image_keys': ['front'],
            'state_keys': ['state'],
            'persistent': 1
        }, TypeError, 'bool'),
        ({
            'image_keys': ['front'],
            'state_keys': ['state'],
            'image_encoding': 'mono8'
        }, ValueError, 'image_encoding'),
    ],
)
def test_transport_validation(transport, exception, message):
    with pytest.raises(exception, match=message):
        FluxVLAROSModelClient(transport=transport)


@pytest.mark.parametrize(
    'mutate_observation,context,exception,message',
    [
        (lambda obs: obs.pop('front'), {
            'prompt': 'task'
        }, KeyError, 'front'),
        (lambda obs: obs.__setitem__('front',
                                     np.zeros((2, 2, 3), dtype=np.float32)), {
                                         'prompt': 'task'
                                     }, TypeError, 'uint8'),
        (lambda obs: obs.__setitem__('front', np.zeros(
            (2, 2), dtype=np.uint8)), {
                'prompt': 'task'
            }, ValueError, r'\[H, W, 3\]'),
        (lambda obs: obs.__setitem__('proprio',
                                     np.asarray([1, 2], dtype=np.int32)), {
                                         'prompt': 'task'
                                     }, TypeError, 'floating dtype'),
        (lambda obs: obs.__setitem__('proprio',
                                     np.asarray([[1.0]], dtype=np.float32)), {
                                         'prompt': 'task'
                                     }, ValueError, r'\[D\]'),
        (lambda obs: obs.__setitem__('proprio',
                                     np.asarray([np.nan], dtype=np.float32)), {
                                         'prompt': 'task'
                                     }, ValueError, 'NaN'),
        (lambda obs: None, {
            'prompt': ''
        }, ValueError, 'prompt'),
    ],
)
def test_predict_rejects_invalid_observations_before_service_call(
        fake_ros, mutate_observation, context, exception, message):
    client = _ready_client(fake_ros)
    observation = _observation()
    mutate_observation(observation)

    with pytest.raises(exception, match=message):
        client.predict_action(observation, context)

    assert fake_ros.requests == []


@pytest.mark.parametrize(
    'mutate_response,exception,message',
    [
        (lambda response: setattr(response, 'request_id', 'wrong'),
         RuntimeError, 'request_id mismatch'),
        (lambda response: (setattr(response, 'ok', False),
                           setattr(response, 'error', 'model failed')),
         RuntimeError, 'model failed'),
        (lambda response: setattr(response, 'denormalized', False),
         RuntimeError, 'normalized actions'),
        (lambda response: setattr(response, 'action_horizon', 0), ValueError,
         'action_horizon must be positive'),
        (lambda response: setattr(response, 'actions', [1.0]), ValueError,
         'action size'),
        (lambda response: setattr(response, 'actions',
                                  [np.nan, 2.0, 3.0, 4.0, 5.0, 6.0]),
         ValueError, 'NaN'),
        (lambda response: setattr(response, 'inference_time_s', -1.0),
         ValueError, 'non-negative'),
    ],
)
def test_predict_strictly_validates_service_responses(fake_ros,
                                                      mutate_response,
                                                      exception, message):

    def responder(request):
        response = fake_ros.valid_response(request)
        mutate_response(response)
        return response

    fake_ros.responder = responder
    client = _ready_client(fake_ros)
    with pytest.raises(exception, match=message):
        client.predict_action(_observation(), {'prompt': 'task'})


def test_failed_response_keeps_first_request_reset_flag(fake_ros):
    response_count = 0

    def responder(request):
        nonlocal response_count
        response_count += 1
        response = fake_ros.valid_response(request)
        if response_count == 1:
            response.request_id = 'stale-request'
        return response

    fake_ros.responder = responder
    client = _ready_client(fake_ros)
    with pytest.raises(RuntimeError, match='request_id mismatch'):
        client.predict_action(_observation(), {'prompt': 'task'})
    client.predict_action(_observation(), {'prompt': 'task'})

    assert [request.reset for request in fake_ros.requests] == [True, True]


def test_predict_requires_setup_and_reset(fake_ros):
    client = FluxVLAROSModelClient(transport=_transport())
    client.reset(TaskSpec(task_id='task', description='task'), 0, 1)
    with pytest.raises(RuntimeError, match=r'setup\(\)'):
        client.predict_action(_observation(), {'prompt': 'task'})

    client = FluxVLAROSModelClient(transport=_transport())
    client.setup()
    with pytest.raises(RuntimeError, match=r'reset\(\)'):
        client.predict_action(_observation(), {'prompt': 'task'})


def test_raw_image_bridge_serializes_noncontiguous_images():
    source = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    image = source[:, ::2, :]

    message = _RawImageBridge(_Image).cv2_to_imgmsg(image, encoding='bgr8')

    assert (message.height, message.width, message.step) == (4, 3, 9)
    assert message.encoding == 'bgr8'
    np.testing.assert_array_equal(
        np.frombuffer(message.data, dtype=np.uint8).reshape(4, 3, 3),
        image,
    )
