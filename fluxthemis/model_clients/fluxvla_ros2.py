from __future__ import annotations

import importlib
import uuid
from typing import Any, Mapping

from fluxthemis.engines.utils import MODEL_CLIENTS

from .fluxvla_ros import (DEFAULT_NODE_NAME, DEFAULT_TIMEOUT_S,
                          FluxVLAROSModelClient)


@MODEL_CLIENTS.register_module()
class FluxVLAROS2ModelClient(FluxVLAROSModelClient):
    """Native ROS 2 service client for a remote FluxVLA policy.

    The wire contract and validation rules intentionally match the ROS 1
    client. ROS imports remain lazy so the rest of FluxThemis can be used
    without sourcing a ROS 2 workspace.
    """

    def __init__(self, transport: Mapping[str, Any]) -> None:
        if not isinstance(transport, Mapping):
            raise TypeError('transport must be a mapping')
        values = dict(transport)
        request_timeout_s = values.pop(
            'request_timeout_s', values.get('timeout_s', DEFAULT_TIMEOUT_S))
        super().__init__(transport=values)
        self.request_timeout_s = self._positive_timeout(
            request_timeout_s, 'transport.request_timeout_s')
        self.transport.pop('persistent', None)
        self.transport['request_timeout_s'] = self.request_timeout_s
        self._rclpy = None
        self._node = None
        self._owns_context = False

    def setup(self) -> None:
        with self._lock:
            if self._service is not None:
                return
            try:
                rclpy = importlib.import_module('rclpy')
                cv_bridge = importlib.import_module('cv_bridge')
                service_module = importlib.import_module('fluxthemis_msgs.srv')
                bridge_type = getattr(cv_bridge, 'CvBridge')
                service_type = getattr(service_module, 'PredictAction')
                request_type = getattr(service_type, 'Request')
                if self.evaluation_reporting_enabled:
                    report_service_type = getattr(service_module,
                                                  'ReportEvaluation')
                    report_request_type = getattr(report_service_type,
                                                  'Request')
                else:
                    report_service_type = None
                    report_request_type = None
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    'FluxVLAROS2ModelClient requires ROS 2, cv_bridge and '
                    'the rosidl-built fluxthemis_msgs/PredictAction service, '
                    'plus ReportEvaluation when reporting is configured. '
                    'Source the ROS 2 installation and message workspace '
                    'before starting evaluation.') from exc

            owns_context = False
            node = None
            service = None
            report_service = None
            try:
                if not rclpy.ok():
                    rclpy.init(args=None)
                    owns_context = True
                node = rclpy.create_node(DEFAULT_NODE_NAME)
                service = node.create_client(service_type, self.service_name)
                available = service.wait_for_service(
                    timeout_sec=self.timeout_s)
                if not available:
                    raise TimeoutError(
                        f'ROS 2 service {self.service_name!r} was not '
                        f'available within {self.timeout_s:g} seconds')
                if self.evaluation_reporting_enabled:
                    report_service = node.create_client(
                        report_service_type, self.report_service_name)
                    report_available = report_service.wait_for_service(
                        timeout_sec=self.timeout_s)
                    if not report_available:
                        raise TimeoutError(
                            f'ROS 2 service {self.report_service_name!r} was '
                            f'not available within {self.timeout_s:g} seconds')
                bridge = bridge_type()
            except BaseException:
                if node is not None:
                    if service is not None:
                        node.destroy_client(service)
                    if report_service is not None:
                        node.destroy_client(report_service)
                    node.destroy_node()
                if owns_context:
                    self._shutdown_context(rclpy)
                raise

            self._rclpy = rclpy
            self._node = node
            self._owns_context = owns_context
            self._bridge = bridge
            self._request_type = request_type
            self._service = service
            self._report_request_type = report_request_type
            self._report_service = report_service
            if self.evaluation_reporting_enabled:
                self._report_run_id = uuid.uuid4().hex
                self._report_sequence = 1
                self._report_run_dir = None

    def close(self) -> None:
        with self._lock:
            rclpy = self._rclpy
            node = self._node
            service = self._service
            report_service = self._report_service
            owns_context = self._owns_context
            self._service = None
            self._request_type = None
            self._report_service = None
            self._report_request_type = None
            self._bridge = None
            self._rclpy = None
            self._node = None
            self._owns_context = False
            self._task = None
            self._episode_index = None
            self._episode_id = None
            self._seed = None
            self._is_new_episode = True

            if node is not None:
                if service is not None:
                    node.destroy_client(service)
                if report_service is not None:
                    node.destroy_client(report_service)
                node.destroy_node()
            if owns_context and rclpy is not None:
                self._shutdown_context(rclpy)

    def _ensure_ready(self) -> None:
        if (self._service is None or self._request_type is None
                or self._bridge is None or self._rclpy is None
                or self._node is None):
            raise RuntimeError('FluxVLAROS2ModelClient.setup() was not called')
        episode_is_unset = any(
            value is None
            for value in (self._task, self._episode_id, self._seed))
        if episode_is_unset:
            raise RuntimeError(
                'FluxVLAROS2ModelClient.reset() must be called before '
                'predict_action()')

    def _ensure_reporting_ready(self) -> None:
        if (self._report_service is None or self._report_request_type is None
                or self._rclpy is None or self._node is None
                or self._report_run_id is None):
            raise RuntimeError(
                'FluxVLAROS2ModelClient.setup() must be called before '
                'report_evaluation()')

    def _build_request(self, request_id: str, observation: Mapping[str, Any],
                       context: Mapping[str, Any]) -> Any:
        stamp = self._node.get_clock().now().to_msg()
        return self._build_request_with_stamp(
            request_id=request_id,
            observation=observation,
            context=context,
            stamp=stamp,
        )

    def _build_report_request(self, request_id: str, sequence: int,
                              event_type: str, payload_json: str) -> Any:
        request = self._report_request_type()
        request.header.stamp = self._node.get_clock().now().to_msg()
        request.request_id = request_id
        request.run_id = self._report_run_id
        request.sequence = sequence
        request.event_type = event_type
        request.payload_json = payload_json
        return request

    def _call_service(self, request: Any) -> Any:
        return self._call_async_service(self._service, request,
                                        self.service_name)

    def _call_report_service_once(self, request: Any) -> Any:
        return self._call_async_service(self._report_service, request,
                                        self.report_service_name)

    def _call_async_service(self, service: Any, request: Any,
                            service_name: str) -> Any:
        future = service.call_async(request)
        self._rclpy.spin_until_future_complete(
            self._node,
            future,
            timeout_sec=self.request_timeout_s,
        )
        if not future.done():
            remove_pending = getattr(service, 'remove_pending_request', None)
            if callable(remove_pending):
                remove_pending(future)
            cancel = getattr(future, 'cancel', None)
            if callable(cancel):
                cancel()
            raise TimeoutError(
                f'ROS 2 service {service_name!r} did not respond '
                f'within {self.request_timeout_s:g} seconds')
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(
                f'ROS 2 service {service_name!r} call failed') \
                from exception
        response = future.result()
        if response is None:
            raise RuntimeError(
                f'ROS 2 service {service_name!r} returned no response')
        return response

    @staticmethod
    def _shutdown_context(rclpy: Any) -> None:
        if not rclpy.ok():
            return
        try_shutdown = getattr(rclpy, 'try_shutdown', None)
        if callable(try_shutdown):
            try_shutdown()
        else:
            rclpy.shutdown()
