"""RoboDojo-specific source, package and generated robot-path checks."""

import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def check_environment(root):
    """Inspect packages, source and asset paths without starting Isaac Sim."""
    root = Path(root).resolve(strict=True)
    for module in ('zmq', 'msgpack', 'mmengine', 'gymnasium', 'fluxthemis'):
        importlib.import_module(module)
    packages = {
        name: importlib.metadata.version(name)
        for name in ('torch', 'isaacsim', 'isaaclab', 'numpy', 'mmengine',
                     'pyzmq', 'msgpack', 'gymnasium')
    }
    required = ('env/global_configs.py', 'env_cfg/arx_x5.yml',
                'src/eval_client/eval_env.py',
                'src/eval_client/physx_warning_monitor.py',
                'task/RoboDojo/task_registry.py', 'Assets/Robots',
                'Assets/Object', 'Assets/Material', 'Assets/Eval_Layout')
    missing = [
        str(root / name) for name in required if not (root / name).exists()
    ]
    if missing:
        raise ValueError('Missing RoboDojo files: ' + ', '.join(missing))
    templates = list((root / 'Assets/Robots').rglob('*_tmp.yml'))
    if not templates:
        raise ValueError('No robot path templates in Assets/Robots')
    for template in templates:
        expected = template.read_text().replace('${ASSETS_PATH}', str(root))
        expected = expected.replace('$ASSETS_PATH', str(root))
        target = template.with_name(template.name.replace('_tmp.yml', '.yml'))
        if not target.is_file() or target.read_text() != expected:
            raise ValueError(f'Robot paths need preparation: {target}; run '
                             'utils/update_embodiment_config_path.py in '
                             f'{root} before evaluation')
    source = {}
    for folder in ('env', 'env_cfg', 'src', 'task', 'utils'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yml', '.yaml'):
                source[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_text().replace(
                        str(root), '${ROBODOJO_ROOT}').encode()).hexdigest()
    installed = sorted((d.metadata['Name'], d.version)
                       for d in importlib.metadata.distributions()
                       if d.metadata['Name'])
    fingerprint = dict(python=sys.version, packages=installed, source=source)
    return dict(
        status='ok',
        packages=packages,
        fingerprint=hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
        coverage='Python version, installed package versions, '
        'RoboDojo env/env_cfg/src/task/utils sources; '
        'not CUDA driver or all installed binary contents')
