from __future__ import annotations

import importlib
import json
import math
import threading
import time
import uuid
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np

from fluxthemis.engines.utils import MODEL_CLIENTS
from fluxthemis.structures import ActionChunk, TaskSpec

from .base import BaseModelClient

DEFAULT_SERVICE_NAME = '/fluxvla/predict_action'
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_UNNORM_KEY = ''
DEFAULT_IMAGE_ENCODING = 'rgb8'
DEFAULT_PERSISTENT = True
DEFAULT_NODE_NAME = 'fluxthemis_eval'
REPORT_MAX_ATTEMPTS = 3
REPORT_RETRY_DELAY_S = 0.1
REPORT_EVENT_TYPES = frozenset(
    {'run_start', 'episode_start', 'episode_end', 'run_end'})
MAX_UINT64 = (1 << 64) - 1

TRANSPORT_FIELDS = frozenset({
    'service_name',
    'timeout_s',
    'image_keys',
    'state_keys',
    'unnorm_key',
    'image_encoding',
    'persistent',
    'report_service_name',
})
SUPPORTED_IMAGE_ENCODINGS = frozenset({'rgb8', 'bgr8'})


class _RawImageBridge:
    """Serialize uint8 HWC images without a compiled ``cv_bridge``."""

    def __init__(self, image_type: Any) -> None:
        self._image_type = image_type

    def cv2_to_imgmsg(self, image: np.ndarray, encoding: str) -> Any:
        if encoding not in SUPPORTED_IMAGE_ENCODINGS:
            raise ValueError(
                f'Unsupported ROS image encoding {encoding!r}; expected one '
                f'of {sorted(SUPPORTED_IMAGE_ENCODINGS)}')
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ValueError(
                'ROS images must be uint8 arrays shaped [H, W, 3]')

        image = np.ascontiguousarray(image)
        message = self._image_type()
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = encoding
        message.is_bigendian = 0
        message.step = int(image.shape[1] * image.shape[2])
        message.data = image.tobytes(order='C')
        return message


@MODEL_CLIENTS.register_module()
class FluxVLAROSModelClient(BaseModelClient):
    """ROS1 service client for a remote FluxVLA policy.

    ROS dependencies are imported only by :meth:`setup`, so importing
    FluxThemis remains possible outside a sourced ROS environment. The remote
    service owns model preprocessing and action denormalization; responses
    that do not explicitly declare denormalized actions are rejected.
    """

    def __init__(self, transport: Mapping[str, Any]) -> None:
        if not isinstance(transport, Mapping):
            raise TypeError('transport must be a mapping')
        values = dict(transport)
        unknown_fields = sorted(set(values) - TRANSPORT_FIELDS)
        if unknown_fields:
            raise ValueError(f'Unknown ROS transport fields: {unknown_fields}')
        if 'image_keys' not in values:
            raise ValueError('transport.image_keys is required')
        if 'state_keys' not in values:
            raise ValueError('transport.state_keys is required')

        self.service_name = self._nonempty_string(
            values.get('service_name', DEFAULT_SERVICE_NAME),
            'transport.service_name',
        )
        self.timeout_s = self._positive_timeout(
            values.get('timeout_s', DEFAULT_TIMEOUT_S),
            'transport.timeout_s',
        )
        self.image_keys = self._normalize_keys(values['image_keys'],
                                               'transport.image_keys')
        self.state_keys = self._normalize_keys(values['state_keys'],
                                               'transport.state_keys')
        self.unnorm_key = self._string(
            values.get('unnorm_key', DEFAULT_UNNORM_KEY),
            'transport.unnorm_key',
        )
        self.image_encoding = self._nonempty_string(
            values.get('image_encoding', DEFAULT_IMAGE_ENCODING),
            'transport.image_encoding',
        )
        if self.image_encoding not in SUPPORTED_IMAGE_ENCODINGS:
            raise ValueError('transport.image_encoding must be one of '
                             f'{sorted(SUPPORTED_IMAGE_ENCODINGS)}')
        persistent = values.get('persistent', DEFAULT_PERSISTENT)
        if not isinstance(persistent, bool):
            raise TypeError('transport.persistent must be a bool')
        self.persistent = persistent
        report_service_name = values.get('report_service_name')
        self.report_service_name = (
            None if report_service_name is None else self._nonempty_string(
                report_service_name, 'transport.report_service_name'))
        self.transport = {
            'service_name': self.service_name,
            'timeout_s': self.timeout_s,
            'image_keys': self.image_keys,
            'state_keys': self.state_keys,
            'unnorm_key': self.unnorm_key,
            'image_encoding': self.image_encoding,
            'persistent': self.persistent,
        }
        if self.report_service_name is not None:
            self.transport['report_service_name'] = self.report_service_name

        self._rospy = None
        self._bridge = None
        self._service = None
        self._request_type = None
        self._report_service = None
        self._report_request_type = None
        self._report_run_id: str | None = None
        self._report_sequence = 1
        self._report_run_dir: str | None = None
        self._task: TaskSpec | None = None
        self._episode_index: int | None = None
        self._episode_id: str | None = None
        self._seed: int | None = None
        self._is_new_episode = True
        self._lock = threading.RLock()
        self.last_profile: dict[str, float] = {}

    def setup(self) -> None:
        with self._lock:
            if self._service is not None:
                return
            try:
                rospy = importlib.import_module('rospy')
                message_module = importlib.import_module('sensor_msgs.msg')
                service_module = importlib.import_module('fluxthemis_msgs.srv')
                image_type = getattr(message_module, 'Image')
                service_type = getattr(service_module, 'PredictAction')
                request_type = getattr(service_module, 'PredictActionRequest')
                if self.evaluation_reporting_enabled:
                    report_service_type = getattr(service_module,
                                                  'ReportEvaluation')
                    report_request_type = getattr(service_module,
                                                  'ReportEvaluationRequest')
                else:
                    report_service_type = None
                    report_request_type = None
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    'FluxVLAROSModelClient requires ROS1 sensor_msgs and the '
                    'fluxthemis_msgs/PredictAction service, plus '
                    'ReportEvaluation when reporting is configured. Source '
                    'the ROS1 workspace before starting evaluation.') from exc

            bridge = _RawImageBridge(image_type)
            self._ensure_ros_node(rospy)
            service = None
            report_service = None
            try:
                rospy.wait_for_service(
                    self.service_name, timeout=self.timeout_s)
                service = rospy.ServiceProxy(
                    self.service_name,
                    service_type,
                    persistent=self.persistent,
                )
                if self.evaluation_reporting_enabled:
                    rospy.wait_for_service(
                        self.report_service_name, timeout=self.timeout_s)
                    report_service = rospy.ServiceProxy(
                        self.report_service_name,
                        report_service_type,
                        persistent=self.persistent,
                    )
            except BaseException:
                for proxy in (service, report_service):
                    close = getattr(proxy, 'close', None)
                    if callable(close):
                        close()
                raise
            self._rospy = rospy
            self._bridge = bridge
            self._request_type = request_type
            self._service = service
            self._report_service = report_service
            self._report_request_type = report_request_type
            if self.evaluation_reporting_enabled:
                self._report_run_id = uuid.uuid4().hex
                self._report_sequence = 1
                self._report_run_dir = None

    @property
    def evaluation_reporting_enabled(self) -> bool:
        return self.report_service_name is not None

    @property
    def evaluation_report_run_id(self) -> str | None:
        return self._report_run_id

    @property
    def evaluation_report_run_dir(self) -> str | None:
        return self._report_run_dir

    def report_evaluation(self, event_type: str,
                          payload: Mapping[str, Any]) -> str | None:
        if not self.evaluation_reporting_enabled:
            return None
        if event_type not in REPORT_EVENT_TYPES:
            raise ValueError(
                f'Unsupported evaluation report event type {event_type!r}; '
                f'expected one of {sorted(REPORT_EVENT_TYPES)}')
        if not isinstance(payload, Mapping):
            raise TypeError('evaluation report payload must be a mapping')

        try:
            payload_json = json.dumps(
                dict(payload),
                ensure_ascii=False,
                separators=(',', ':'),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'evaluation report payload must be JSON serializable') from exc

        with self._lock:
            self._ensure_reporting_ready()
            if self._report_sequence > MAX_UINT64:
                raise OverflowError(
                    'evaluation report sequence exhausted uint64')
            request_id = uuid.uuid4().hex
            request = self._build_report_request(
                request_id=request_id,
                sequence=self._report_sequence,
                event_type=event_type,
                payload_json=payload_json,
            )
            response = self._call_report_service_with_retries(request)
            run_dir = self._validate_report_response(response, request_id)
            if (self._report_run_dir is not None
                    and run_dir != self._report_run_dir):
                raise RuntimeError(
                    'FluxVLA report service changed the authoritative run_dir '
                    f'from {self._report_run_dir!r} to {run_dir!r}')
            self._report_run_dir = run_dir
            self._report_sequence += 1
            return run_dir

    def reset(self, task: TaskSpec, episode_index: int, seed: int) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError('task must be a TaskSpec')
        if isinstance(episode_index,
                      bool) or not isinstance(episode_index, Integral):
            raise TypeError('episode_index must be an integer')
        if episode_index < 0:
            raise ValueError('episode_index must be non-negative')
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError('seed must be an integer')
        with self._lock:
            self._task = task
            self._episode_index = int(episode_index)
            self._episode_id = f'{task.task_id}:{int(episode_index)}'
            self._seed = int(seed)
            self._is_new_episode = True

    @property
    def supports_explicit_episode_id(self) -> bool:
        return True

    def set_episode_id(self, episode_id: str) -> None:
        episode_id = self._nonempty_string(episode_id, 'episode_id')
        with self._lock:
            if self._task is None or self._episode_index is None:
                raise RuntimeError(
                    'reset() must be called before set_episode_id()')
            self._episode_id = episode_id
            self._is_new_episode = True

    def predict_action(self, observation: Any,
                       context: Mapping[str, Any]) -> ActionChunk:
        if not isinstance(observation, Mapping):
            raise TypeError('FluxVLA ROS observations must be mappings')
        if not isinstance(context, Mapping):
            raise TypeError('FluxVLA ROS context must be a mapping')

        with self._lock:
            self._ensure_ready()
            started = time.perf_counter()
            request_id = uuid.uuid4().hex
            request = self._build_request(
                request_id=request_id,
                observation=observation,
                context=context,
            )
            response = self._call_service(request)
            roundtrip_s = time.perf_counter() - started
            actions, server_infer_s = self._validate_response(
                response, request_id)
            was_reset = self._is_new_episode
            self._is_new_episode = False

        self.last_profile = {
            'roundtrip_s': roundtrip_s,
            'server_infer_s': server_infer_s,
            'network_and_serialization_s': max(roundtrip_s - server_infer_s,
                                               0.0),
        }
        return ActionChunk(
            actions=actions,
            metadata={
                'profile': dict(self.last_profile),
                'request_id': request_id,
                'episode_id': self._episode_id,
                'reset': was_reset,
                'seed': self._seed,
                'denormalized': True,
            },
        )

    def close(self) -> None:
        with self._lock:
            service = self._service
            report_service = self._report_service
            self._service = None
            self._request_type = None
            self._report_service = None
            self._report_request_type = None
            self._bridge = None
            self._rospy = None
            self._task = None
            self._episode_index = None
            self._episode_id = None
            self._seed = None
            self._is_new_episode = True
            if service is not None:
                close = getattr(service, 'close', None)
                if callable(close):
                    close()
            if report_service is not None:
                close = getattr(report_service, 'close', None)
                if callable(close):
                    close()

    def _ensure_ready(self) -> None:
        if (self._service is None or self._request_type is None
                or self._bridge is None or self._rospy is None):
            raise RuntimeError('FluxVLAROSModelClient.setup() was not called')
        episode_is_unset = any(
            value is None
            for value in (self._task, self._episode_id, self._seed))
        if episode_is_unset:
            raise RuntimeError(
                'FluxVLAROSModelClient.reset() must be called before '
                'predict_action()')

    def _ensure_reporting_ready(self) -> None:
        if (self._report_service is None or self._report_request_type is None
                or self._rospy is None or self._report_run_id is None):
            raise RuntimeError(
                'FluxVLAROSModelClient.setup() must be called before '
                'report_evaluation()')

    def _build_request(self, request_id: str, observation: Mapping[str, Any],
                       context: Mapping[str, Any]) -> Any:
        return self._build_request_with_stamp(
            request_id=request_id,
            observation=observation,
            context=context,
            stamp=self._rospy.Time.now(),
        )

    def _build_report_request(self, request_id: str, sequence: int,
                              event_type: str, payload_json: str) -> Any:
        request = self._report_request_type()
        request.header.stamp = self._rospy.Time.now()
        request.request_id = request_id
        request.run_id = self._report_run_id
        request.sequence = sequence
        request.event_type = event_type
        request.payload_json = payload_json
        return request

    def _build_request_with_stamp(self, request_id: str,
                                  observation: Mapping[str, Any],
                                  context: Mapping[str,
                                                   Any], stamp: Any) -> Any:
        images = []
        for key in self.image_keys:
            if key not in observation:
                raise KeyError(f'Observation is missing image key {key!r}')
            image = self._validate_image(observation[key], key)
            images.append(
                self._bridge.cv2_to_imgmsg(
                    image,
                    encoding=self.image_encoding,
                ))

        state_sizes = []
        state_values = []
        for key in self.state_keys:
            if key not in observation:
                raise KeyError(f'Observation is missing state key {key!r}')
            state = self._validate_state(observation[key], key)
            state_sizes.append(int(state.size))
            state_values.extend(state.tolist())
        prompt = self._resolve_prompt(context)
        unnorm_key = context.get('unnorm_key', self.unnorm_key)
        unnorm_key = self._string(unnorm_key, 'context.unnorm_key')

        request = self._request_type()
        request.header.stamp = stamp
        request.request_id = request_id
        request.episode_id = self._episode_id
        request.reset = self._is_new_episode
        request.seed = self._seed
        request.prompt = prompt
        request.image_names = list(self.image_keys)
        request.images = images
        request.state_names = list(self.state_keys)
        request.state_sizes = state_sizes
        request.state_values = state_values
        request.unnorm_key = unnorm_key
        return request

    def _call_service(self, request: Any) -> Any:
        return self._service(request)

    def _call_report_service_with_retries(self, request: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, REPORT_MAX_ATTEMPTS + 1):
            try:
                return self._call_report_service_once(request)
            except Exception as exc:  # noqa: BLE001 - ROS transport exceptions
                last_error = exc
                if attempt < REPORT_MAX_ATTEMPTS:
                    time.sleep(REPORT_RETRY_DELAY_S * attempt)
        raise RuntimeError('FluxVLA report service call failed after '
                           f'{REPORT_MAX_ATTEMPTS} attempts') from last_error

    def _call_report_service_once(self, request: Any) -> Any:
        return self._report_service(request)

    @staticmethod
    def _validate_report_response(response: Any, request_id: str) -> str:
        response_request_id = getattr(response, 'request_id', None)
        if response_request_id != request_id:
            raise RuntimeError(
                'FluxVLA report response request_id mismatch: expected '
                f'{request_id!r}, got {response_request_id!r}')
        if not bool(getattr(response, 'accepted', False)):
            error = str(getattr(response, 'error', '')).strip()
            raise RuntimeError('FluxVLA report service rejected the event: '
                               f'{error or "unknown error"}')
        run_dir = getattr(response, 'run_dir', None)
        if not isinstance(run_dir, str) or not run_dir.strip():
            raise RuntimeError(
                'FluxVLA report service returned an empty run_dir')
        return run_dir.strip()

    def _resolve_prompt(self, context: Mapping[str, Any]) -> str:
        if 'prompt' in context:
            prompt = context['prompt']
        elif 'task_description' in context:
            prompt = context['task_description']
        else:
            prompt = self._task.description
        prompt = self._nonempty_string(prompt, 'policy prompt')
        return prompt

    @staticmethod
    def _validate_image(value: Any, key: str) -> np.ndarray:
        value = FluxVLAROSModelClient._as_numpy_if_tensor(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(f'Observation image {key!r} must be a numpy array')
        if value.dtype != np.uint8:
            raise TypeError(
                f'Observation image {key!r} must have dtype uint8, got '
                f'{value.dtype}')
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(
                f'Observation image {key!r} must have shape [H, W, 3], got '
                f'{value.shape}')
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError(f'Observation image {key!r} cannot be empty')
        return np.ascontiguousarray(value)

    @staticmethod
    def _validate_state(value: Any, key: str) -> np.ndarray:
        value = FluxVLAROSModelClient._as_numpy_if_tensor(value)
        state = np.asarray(value)
        if not np.issubdtype(state.dtype, np.floating):
            raise TypeError(
                f'Observation state {key!r} must have a floating dtype, got '
                f'{state.dtype}')
        if state.ndim != 1:
            raise ValueError(
                f'Observation state {key!r} must have shape [D], got '
                f'{state.shape}')
        if state.size == 0:
            raise ValueError(f'Observation state {key!r} cannot be empty')
        if not np.isfinite(state).all():
            raise ValueError(
                f'Observation state {key!r} contains NaN or infinity')
        return np.asarray(state, dtype=np.float32)

    @staticmethod
    def _validate_response(response: Any,
                           request_id: str) -> tuple[np.ndarray, float]:
        response_request_id = getattr(response, 'request_id', None)
        if response_request_id != request_id:
            raise RuntimeError(
                f'FluxVLA ROS response request_id mismatch: expected '
                f'{request_id!r}, got {response_request_id!r}')
        if not bool(getattr(response, 'ok', False)):
            error = str(getattr(response, 'error', '')).strip()
            raise RuntimeError(
                f'FluxVLA ROS service error: {error or "unknown error"}')
        if not bool(getattr(response, 'denormalized', False)):
            raise RuntimeError(
                'FluxVLA ROS service returned normalized actions; '
                'environment-unit actions are required')

        horizon = FluxVLAROSModelClient._positive_dimension(
            getattr(response, 'action_horizon', None), 'action_horizon')
        action_dim = FluxVLAROSModelClient._positive_dimension(
            getattr(response, 'action_dim', None), 'action_dim')
        flat_actions = np.asarray(getattr(response, 'actions', None))
        if flat_actions.ndim != 1:
            raise ValueError(
                f'FluxVLA ROS response actions must be flat, got shape '
                f'{flat_actions.shape}')
        expected_size = horizon * action_dim
        if flat_actions.size != expected_size:
            raise ValueError(
                f'FluxVLA ROS response action size {flat_actions.size} does '
                f'not match action_horizon*action_dim={expected_size}')
        try:
            actions = np.asarray(
                flat_actions, dtype=np.float32).reshape(horizon, action_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'FluxVLA ROS response actions must be numeric') from exc
        if not np.isfinite(actions).all():
            raise ValueError(
                'FluxVLA ROS service returned NaN or infinite actions')

        try:
            server_infer_s = float(getattr(response, 'inference_time_s', 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError('inference_time_s must be numeric') from exc
        if not math.isfinite(server_infer_s) or server_infer_s < 0:
            raise ValueError(
                'inference_time_s must be finite and non-negative')
        return actions, server_infer_s

    @staticmethod
    def _positive_dimension(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f'{name} must be an integer')
        value = int(value)
        if value <= 0:
            raise ValueError(f'{name} must be positive')
        return value

    @staticmethod
    def _normalize_keys(value: Any, name: str) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f'{name} must be a sequence')
        keys = tuple(value)
        if not keys:
            raise ValueError(f'{name} cannot be empty')
        for key in keys:
            FluxVLAROSModelClient._nonempty_string(key, f'{name} entry')
        if len(keys) != len(set(keys)):
            raise ValueError(f'{name} cannot contain duplicates')
        return keys

    @staticmethod
    def _ensure_ros_node(rospy: Any) -> None:
        core = getattr(rospy, 'core', None)
        is_initialized = (
            core is not None
            and callable(getattr(core, 'is_initialized', None))
            and core.is_initialized())
        if not is_initialized:
            rospy.init_node(
                DEFAULT_NODE_NAME, anonymous=True, disable_signals=True)

    @staticmethod
    def _positive_timeout(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'{name} must be numeric')
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return value

    @staticmethod
    def _string(value: Any, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f'{name} must be a string')
        return value

    @staticmethod
    def _nonempty_string(value: Any, name: str) -> str:
        value = FluxVLAROSModelClient._string(value, name)
        if not value:
            raise ValueError(f'{name} cannot be empty')
        return value

    @staticmethod
    def _as_numpy_if_tensor(value: Any) -> Any:
        if hasattr(value, 'detach') and hasattr(value, 'cpu'):
            return value.detach().cpu().numpy()
        return value
