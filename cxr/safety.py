"""Prevent generated artifacts from overwriting raw data or source inputs."""
from pathlib import Path


def safe_output(path, data_root=None, inputs=()):
    target = Path(path).resolve()
    roots = [Path(__file__).resolve().parents[1] / 'cxr_first']
    if data_root is not None:
        roots.append(Path(data_root))
    for root in roots:
        root = root.resolve()
        if target == root or target.is_relative_to(root):
            raise ValueError(f'Refusing output inside raw data: {target}')
    for source in inputs:
        if target == Path(source).resolve():
            raise ValueError(f'Output would overwrite input: {target}')
    return target


def fresh_run_directory(path, data_root):
    target = safe_output(path, data_root)
    if target.exists() and any(target.iterdir()):
        raise ValueError(f'Run directory is not empty; choose a new --output: {target}')
    target.mkdir(parents=True, exist_ok=True)
    return target
