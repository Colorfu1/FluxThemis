"""Adapt Themis launch options to FluxVLA's existing model entry point."""

import argparse
import importlib
import os
import runpy
import sys
from pathlib import Path


def scheduling_arguments(arguments, scheduling):
    """Explicit scheduling overrides native cfg-options."""
    arguments = list(arguments)
    if scheduling is None:
        return arguments
    # argparse accepts --cfg-options=KEY=VALUE as a single-value option.
    # Split its flag before appending our additional assignment.
    arguments = [
        part for argument in arguments
        for part in (argument.split('=', 1) if argument.
                     startswith('--cfg-options=') else [argument])
    ]
    value = ('themis.server.strict_worker_assignment=' +
             str(scheduling == 'fixed'))
    positions = [
        i for i, argument in enumerate(arguments)
        if argument == '--cfg-options'
    ]
    if not positions:
        return arguments + ['--cfg-options', value]
    end = positions[-1] + 1
    while end < len(arguments) and not arguments[end].startswith('--'):
        end += 1
    arguments.insert(end, value)
    return arguments


def replace_option(arguments, flag, value):
    """Replace all occurrences with the normalized final value."""
    result = []
    args = iter(arguments)
    for argument in args:
        if argument == flag:
            next(args)
        elif not argument.startswith(flag + '='):
            result.append(argument)
    return [*result, flag, value]


def validate_model_config(args):
    """Use FluxVLA's config/checkpoint rules without its CUDA entry point."""
    from fluxvla.engines.runners.serving.serving_runner import (
        config_get, get_server_config, require_mapping,
        resolve_checkpoint_path)
    from mmengine import Config

    cfg = Config.fromfile(args.config, import_custom_modules=False)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    themis = require_mapping(config_get(cfg, 'themis'), 'config.themis')
    server = get_server_config(themis)
    section_name = server.get('dataset_section', 'eval')
    if section_name not in ('eval', 'inference'):
        raise ValueError('dataset_section must be eval or inference')
    section = require_mapping(
        config_get(cfg, section_name), f'config.{section_name}')
    resolve_checkpoint_path(
        args.ckpt_path or server.get('ckpt_path') or section.get('ckpt_path'))
    for dependency in ('zmq', 'msgpack'):
        importlib.import_module(dependency)
    print('Distributed model configuration validated.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        '--fluxvla-root',
        type=Path,
        default=os.environ.get('FLUXVLA_ROOT',
                               Path(__file__).resolve().parents[2] /
                               'FluxVLA'))
    parser.add_argument('--scheduling', choices=('dynamic', 'fixed'))
    parser.add_argument('--dry-run', action='store_true')
    options, arguments = parser.parse_known_args(argv)
    root = options.fluxvla_root.expanduser().resolve()
    entry_path = root / 'scripts/distributed_zmq_server.py'
    if not entry_path.is_file():
        parser.error(f'FluxVLA model entry point not found: {entry_path}')
    sys.path.insert(0, str(root))
    entry = runpy.run_path(str(entry_path))
    if '-h' in arguments or '--help' in arguments:
        print(
            'Themis model launcher options: --fluxvla-root PATH, '
            '--scheduling dynamic|fixed, --dry-run',
            flush=True)
    arguments = scheduling_arguments(arguments, options.scheduling)
    args = entry['parse_args'](arguments)
    if args.workers_per_gpu < 1:
        parser.error('workers-per-gpu must be positive')
    # CLI paths are caller-relative; paths inside config remain VLA-relative.
    args.config = str(Path(args.config).expanduser().resolve(strict=True))
    arguments = replace_option(arguments, '--config', args.config)
    if args.ckpt_path:
        args.ckpt_path = str(Path(args.ckpt_path).expanduser().resolve())
        arguments = replace_option(arguments, '--ckpt-path', args.ckpt_path)
    os.chdir(root)
    if options.dry_run:
        validate_model_config(args)
        return 0
    # Native main retains rank-to-device mapping, model building and serving.
    # In particular, do not initialize CUDA or rewrite LOCAL_RANK here.
    return entry['main'](arguments)


if __name__ == '__main__':
    raise SystemExit(main())
