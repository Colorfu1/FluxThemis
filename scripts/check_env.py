#!/usr/bin/env python3
"""Validate one activated FluxThemis runtime without changing it."""

from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

REQUEST_FIELDS = (
    'header',
    'request_id',
    'episode_id',
    'reset',
    'seed',
    'prompt',
    'image_names',
    'images',
    'state_names',
    'state_sizes',
    'state_values',
    'unnorm_key',
)
RESPONSE_FIELDS = (
    'header',
    'request_id',
    'ok',
    'error',
    'actions',
    'action_horizon',
    'action_dim',
    'denormalized',
    'inference_time_s',
)
REPORT_REQUEST_FIELDS = (
    'header',
    'request_id',
    'run_id',
    'sequence',
    'event_type',
    'payload_json',
)
REPORT_RESPONSE_FIELDS = (
    'header',
    'request_id',
    'accepted',
    'error',
    'run_dir',
)


class CheckError(RuntimeError):
    """Raised when the activated runtime violates its deployment contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate the currently activated FluxThemis runtime.')
    parser.add_argument('--ros', choices=('1', '2', 'none'), required=True)
    parser.add_argument(
        '--libero', choices=('imports', 'none'), default='none')
    parser.add_argument('--skip-pip-check', action='store_true')
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def pass_check(name: str, detail: str) -> None:
    print(f'[PASS] {name}: {detail}')


def check_python(ros_version: str) -> None:
    version = sys.version_info[:2]
    if ros_version == '1':
        require(version == (3, 10),
                f'ROS 1 runtime requires Python 3.10, found {version}')
    elif ros_version == '2':
        require(version == (3, 12),
                f'ROS 2 Jazzy runtime requires Python 3.12, found {version}')
    else:
        require(version in {(3, 10), (3, 12)},
                f'FluxThemis supports Python 3.10 or 3.12, found {version}')
    pass_check('python', sys.version.split()[0])


def check_fluxthemis() -> None:
    import fluxthemis
    from fluxthemis.engines import MODEL_CLIENTS

    require(
        MODEL_CLIENTS.get('FluxVLAROSModelClient') is not None,
        'FluxVLAROSModelClient is not registered')
    require(
        MODEL_CLIENTS.get('FluxVLAROS2ModelClient') is not None,
        'FluxVLAROS2ModelClient is not registered')
    package_path = Path(inspect.getfile(fluxthemis)).resolve()
    pass_check('fluxthemis', str(package_path))


def require_environment(name: str, expected: str) -> None:
    actual = os.environ.get(name)
    require(actual == expected,
            f'{name} must be {expected!r}, found {actual!r}')


def reject_path_fragment(fragment: str, variable_names: tuple[str,
                                                              ...]) -> None:
    for name in variable_names:
        value = os.environ.get(name, '')
        require(fragment not in value,
                f'{name} contains the other ROS stack: {value!r}')


def check_ros1() -> None:
    import rospy
    from fluxthemis_msgs.srv import (PredictAction, PredictActionRequest,
                                     PredictActionResponse, ReportEvaluation,
                                     ReportEvaluationRequest,
                                     ReportEvaluationResponse)
    from sensor_msgs.msg import Image

    from fluxthemis.model_clients.fluxvla_ros import FluxVLAROSModelClient

    require_environment('ROS_VERSION', '1')
    require_environment('ROS_DISTRO', 'noetic')
    require('FLUXTHEMIS_ROS2_WORKSPACE' not in os.environ,
            'ROS 2 workspace leaked into the ROS 1 process')
    workspace = Path(
        os.environ.get('FLUXTHEMIS_ROS1_WORKSPACE',
                       '/root/fluxthemis_ros1_ws')).resolve()
    require((workspace / 'install_isolated/setup.sh').is_file(),
            f'ROS 1 overlay is missing under {workspace}')
    reject_path_fragment(
        '/opt/ros/jazzy',
        ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH'),
    )
    require(
        tuple(PredictActionRequest.__slots__) == REQUEST_FIELDS,
        'loaded ROS 1 request type has stale or incompatible fields')
    require(
        tuple(PredictActionResponse.__slots__) == RESPONSE_FIELDS,
        'loaded ROS 1 response type has stale or incompatible fields')
    require(PredictAction._type == 'fluxthemis_msgs/PredictAction',
            f'unexpected ROS 1 service type: {PredictAction._type!r}')
    require(
        tuple(ReportEvaluationRequest.__slots__) == REPORT_REQUEST_FIELDS,
        'loaded ROS 1 report request type has stale or incompatible fields')
    require(
        tuple(ReportEvaluationResponse.__slots__) == REPORT_RESPONSE_FIELDS,
        'loaded ROS 1 report response type has stale or incompatible fields')
    require(
        ReportEvaluation._type == 'fluxthemis_msgs/ReportEvaluation',
        f'unexpected ROS 1 report service type: {ReportEvaluation._type!r}')
    require(Image._type == 'sensor_msgs/Image',
            f'unexpected image type: {Image._type!r}')
    pass_check(
        'ros1',
        f'Noetic at {workspace} ({rospy.__name__}, '
        f'{FluxVLAROSModelClient.__name__})',
    )


def check_ros2() -> None:
    import cv_bridge
    import lark
    import launch
    import rclpy
    from fluxthemis_msgs.srv import PredictAction, ReportEvaluation

    from fluxthemis.model_clients.fluxvla_ros2 import FluxVLAROS2ModelClient

    require_environment('ROS_VERSION', '2')
    require_environment('ROS_DISTRO', 'jazzy')
    require('FLUXTHEMIS_ROS1_WORKSPACE' not in os.environ,
            'ROS 1 workspace leaked into the ROS 2 process')
    workspace = Path(
        os.environ.get('FLUXTHEMIS_ROS2_WORKSPACE',
                       '/root/fluxthemis_ros2_ws')).resolve()
    require((workspace / 'install/setup.sh').is_file(),
            f'ROS 2 overlay is missing under {workspace}')
    reject_path_fragment(
        'fluxthemis_ros1_ws',
        ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH'),
    )
    request_fields = tuple(
        PredictAction.Request.get_fields_and_field_types().keys())
    response_fields = tuple(
        PredictAction.Response.get_fields_and_field_types().keys())
    require(request_fields == REQUEST_FIELDS,
            'loaded ROS 2 request type has stale or incompatible fields')
    require(response_fields == RESPONSE_FIELDS,
            'loaded ROS 2 response type has stale or incompatible fields')
    report_request_fields = tuple(
        ReportEvaluation.Request.get_fields_and_field_types().keys())
    report_response_fields = tuple(
        ReportEvaluation.Response.get_fields_and_field_types().keys())
    require(
        report_request_fields == REPORT_REQUEST_FIELDS,
        'loaded ROS 2 report request type has stale or incompatible fields')
    require(
        report_response_fields == REPORT_RESPONSE_FIELDS,
        'loaded ROS 2 report response type has stale or incompatible fields')
    pass_check(
        'ros2',
        f'Jazzy at {workspace} ({rclpy.__name__}, {cv_bridge.__name__}, '
        f'{launch.__name__}, lark {lark.__version__}, '
        f'{FluxVLAROS2ModelClient.__name__})',
    )


def check_libero() -> None:
    import robosuite
    import torch

    from fluxthemis.environments.sim.libero import _load_libero_runtime

    runtime = _load_libero_runtime()
    libero_root = Path(inspect.getfile(runtime.libero)).resolve().parent
    for name in ('assets', 'bddl_files', 'init_states'):
        path = Path(runtime.libero.get_libero_path(name)).resolve()
        require(path.is_dir(), f'LIBERO {name} path does not exist: {path}')
        try:
            path.relative_to(libero_root)
        except ValueError as exc:
            raise CheckError(
                f'LIBERO {name} resolves outside {libero_root}: {path}') \
                from exc
    pass_check(
        'libero',
        f'{libero_root} (robosuite {robosuite.__version__}, '
        f'torch {torch.__version__})',
    )


def check_pip() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'check'],
        check=False,
        capture_output=True,
        text=True,
    )
    output = '\n'.join(part.strip() for part in (result.stdout, result.stderr)
                       if part.strip())
    require(result.returncode == 0, f'pip check failed:\n{output}')
    pass_check('pip', output or 'no broken requirements')


def main() -> int:
    args = parse_args()
    checks: list[Callable[[], None]] = [
        lambda: check_python(args.ros),
        check_fluxthemis,
    ]
    if args.ros == '1':
        checks.append(check_ros1)
    elif args.ros == '2':
        checks.append(check_ros2)
    if args.libero == 'imports':
        checks.append(check_libero)
    if not args.skip_pip_check:
        checks.append(check_pip)

    try:
        for check in checks:
            check()
    except Exception as exc:
        print(f'[FAIL] {exc}', file=sys.stderr)
        return 1

    print(f'Environment check passed (ROS {args.ros}).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
