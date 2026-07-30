from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path

from mmengine import Config, DictAction
from mmengine.utils import import_modules_from_strings

from fluxthemis.engines import build_runner_from_cfg

DEFAULT_ARTIFACT_SUBDIR = 'fluxthemis'
RUN_ARTIFACT_FILENAMES = (
    'summary.json',
    'episodes.jsonl',
    'progress.json',
    'failure.json',
    'resolved_config.py',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate a policy on FluxThemis environments.')
    parser.add_argument(
        '--config',
        required=True,
        help=('Path to the FluxVLA MMEngine config containing a `themis` '
              'section.'),
    )
    parser.add_argument(
        '--work-dir',
        default=None,
        help=('Artifact directory inside the FluxVLA `work_dirs` tree. '
              'Relative paths are resolved from FluxVLA/work_dirs.'),
    )
    parser.add_argument('--run-name', default=None)
    parser.add_argument(
        '--parallel-workers',
        type=int,
        default=None,
        help=('Number of spawned simulator rollout workers. Overrides '
              '`themis.runner.parallel_workers`.'),
    )
    parser.add_argument(
        '--simulator-gpus',
        default=None,
        help=('Comma-separated physical GPU ids used for simulator rendering. '
              'Ids are assigned round-robin to rollout workers.'),
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=('Override config values, for example '
              'themis.runner.seed=42.'),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build registered components without allocating resources.',
    )
    return parser.parse_args(argv)


def load_config(args) -> tuple[Config, object]:
    config_path = Path(args.config).expanduser().resolve(strict=True)
    fluxvla_root = _find_fluxvla_root(config_path)
    # FluxVLA top-level custom imports may require its full model environment.
    # This process imports only the explicitly scoped Themis plugins below.
    cfg = Config.fromfile(str(config_path), import_custom_modules=False)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if 'themis' not in cfg:
        raise KeyError(
            'FluxVLA config must define a `themis` section; FluxThemis no '
            'longer owns a separate evaluation config.')
    if not isinstance(cfg.themis, Mapping):
        raise TypeError('FluxVLA config `themis` must be a mapping')

    custom_imports = cfg.themis.get('custom_imports')
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    if 'runner' not in cfg.themis:
        raise KeyError('FluxVLA config `themis` must define `runner`')
    if not isinstance(cfg.themis.runner, Mapping):
        raise TypeError('FluxVLA config `themis.runner` must be a mapping')

    runner_cfg = copy.deepcopy(cfg.themis.runner)
    requested_work_dir = (
        args.work_dir
        if args.work_dir is not None else runner_cfg.get('work_dir'))
    runner_cfg['work_dir'] = str(
        _resolve_work_dir(fluxvla_root, requested_work_dir))
    if args.run_name is not None:
        runner_cfg['run_name'] = args.run_name
    if args.parallel_workers is not None:
        if args.parallel_workers < 1:
            raise ValueError('--parallel-workers must be positive')
        runner_cfg['parallel_workers'] = args.parallel_workers
    if args.simulator_gpus is not None:
        simulator_gpu_ids = [
            value.strip() for value in args.simulator_gpus.split(',')
            if value.strip()
        ]
        if not simulator_gpu_ids:
            raise ValueError('--simulator-gpus must contain at least one id')
        runner_cfg['simulator_gpu_ids'] = simulator_gpu_ids

    # Persist only the canonical config and CLI overrides. The transport copy
    # below is a build-time injection shared with the FluxVLA ROS server.
    cfg.themis.runner = copy.deepcopy(runner_cfg)
    model_client_cfg = runner_cfg.get('model_client')
    client_ros_version = _ros_model_client_version(model_client_cfg)
    if client_ros_version is not None:
        transport = cfg.themis.get('transport')
        if not isinstance(transport, Mapping):
            raise KeyError('`themis.transport` must be defined for a '
                           'FluxVLA ROS model client')
        if 'transport' in model_client_cfg:
            raise ValueError(
                'Define ROS transport only once at `themis.transport`, not '
                'inside `themis.runner.model_client`.')
        model_client_cfg['transport'] = copy.deepcopy(transport)
        _validate_ros_version_pair(
            cfg.themis.get('ros_server'), client_ros_version)
    return cfg, runner_cfg


def ensure_run_artifacts_available(runner_cfg: Mapping,
                                   dry_run: bool = False) -> None:
    """Reject an explicit run name that would overwrite prior artifacts."""
    if (dry_run or bool(runner_cfg.get('overwrite', False))
            or _remote_reporting_configured(runner_cfg)):
        return
    run_name = runner_cfg.get('run_name')
    if not run_name:
        return
    if (not isinstance(run_name, str) or run_name in {'.', '..'}
            or Path(run_name).name != run_name
            or Path(run_name).is_absolute()):
        raise ValueError('run_name must be a non-empty path component')

    work_dir = Path(runner_cfg['work_dir']).expanduser().resolve()
    run_dir = (work_dir / run_name).resolve()
    if run_dir.parent != work_dir:
        raise ValueError('run_name must stay directly inside work_dir')
    existing = [
        run_dir / filename for filename in RUN_ARTIFACT_FILENAMES
        if (run_dir / filename).exists()
    ]
    if not existing:
        return
    names = ', '.join(path.name for path in existing)
    raise FileExistsError(
        f'Run directory already contains evaluation artifacts: {run_dir}\n'
        f'Existing artifacts: {names}\n'
        'Use a new --run-name to preserve them, or explicitly pass '
        '--cfg-options themis.runner.overwrite=True to replace them.')


def _remote_reporting_configured(runner_cfg: Mapping) -> bool:
    model_client = runner_cfg.get('model_client')
    if not isinstance(model_client, Mapping):
        return False
    transport = model_client.get('transport')
    if not isinstance(transport, Mapping):
        return False
    return bool(transport.get('report_service_name'))


def _ros_model_client_version(model_client_cfg) -> int | None:
    if not isinstance(model_client_cfg, Mapping):
        return None
    client_type = model_client_cfg.get('type')
    if isinstance(client_type, str):
        client_name = client_type.rsplit('.', 1)[-1]
    else:
        client_name = getattr(client_type, '__name__', '')
    return {
        'FluxVLAROSModelClient': 1,
        'FluxVLAROS2ModelClient': 2,
    }.get(client_name)


def _validate_ros_version_pair(server_cfg, client_ros_version: int) -> None:
    if not isinstance(server_cfg, Mapping):
        raise KeyError('`themis.ros_server` must be defined for a FluxVLA '
                       'ROS model client')
    server_ros_version = server_cfg.get('ros_version', 1)
    if isinstance(server_ros_version, bool):
        raise TypeError('`themis.ros_server.ros_version` must be 1 or 2')
    if isinstance(server_ros_version, str):
        server_ros_version = server_ros_version.strip()
    if server_ros_version in (1, '1'):
        server_ros_version = 1
    elif server_ros_version in (2, '2'):
        server_ros_version = 2
    else:
        raise ValueError('`themis.ros_server.ros_version` must be 1 or 2')
    if server_ros_version != client_ros_version:
        raise ValueError(
            f'ROS client/server version mismatch: model client uses ROS '
            f'{client_ros_version}, but themis.ros_server.ros_version is '
            f'{server_ros_version}')


def _find_fluxvla_root(config_path: Path) -> Path:
    for candidate in config_path.parent.parents:
        package_init = candidate / 'fluxvla' / '__init__.py'
        configs_root = candidate / 'configs'
        if not package_init.is_file() or not configs_root.is_dir():
            continue
        resolved_candidate = candidate.resolve()
        try:
            config_path.relative_to(configs_root.resolve())
        except ValueError:
            continue
        return resolved_candidate
    raise ValueError(
        f'Config must be located under a FluxVLA `configs` directory: '
        f'{config_path}')


def _resolve_work_dir(fluxvla_root: Path,
                      value: str | os.PathLike | None) -> Path:
    work_dirs_root = (fluxvla_root / 'work_dirs').resolve()
    if value is None:
        candidate = work_dirs_root / DEFAULT_ARTIFACT_SUBDIR
    else:
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError('themis.runner.work_dir must be a path')
        if not str(value).strip():
            raise ValueError('themis.runner.work_dir cannot be empty')
        requested = Path(value).expanduser()
        if requested.is_absolute():
            candidate = requested
        elif requested.parts and requested.parts[0] == 'work_dirs':
            candidate = fluxvla_root / requested
        else:
            candidate = work_dirs_root / requested

    resolved = candidate.resolve()
    try:
        resolved.relative_to(work_dirs_root)
    except ValueError as exc:
        raise ValueError(
            f'FluxThemis artifacts must stay inside FluxVLA work_dirs: '
            f'{work_dirs_root}') from exc
    return resolved


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg, runner_cfg = load_config(args)
    runner = build_runner_from_cfg(runner_cfg)
    runner.set_resolved_config(
        source=str(Path(args.config).expanduser().resolve()),
        text=cfg.pretty_text,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    'runner': type(runner).__name__,
                    'environment': type(runner.environment).__name__,
                    'model_client': type(runner.model_client).__name__,
                    'evaluator': type(runner.evaluator).__name__,
                },
                indent=2,
            ))
        runner.cleanup()
        return 0

    try:
        runner.run_setup()
        if runner.local_artifacts_enabled:
            cfg.dump(str(runner.resolved_config_path))
        summary = runner.run()
        if runner.local_artifacts_enabled:
            print(json.dumps(summary['metrics'], indent=2))
            print(f'Summary written to {runner.summary_path}')
        else:
            print(f'Evaluation reported to {runner.authoritative_run_dir}')
    finally:
        runner.cleanup()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
