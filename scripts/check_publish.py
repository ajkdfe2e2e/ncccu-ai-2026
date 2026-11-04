"""Fail closed if tracked/staged Git paths contain private data or model artifacts.
Checks Git's index, not just .gitignore; run before committing/pushing. Does not
push anything, inspect credentials, or read patient content.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = {'cxr_first', 'data', 'weights', 'outputs', '.venv', '.cache'}
PRIVATE_SUFFIXES = {'.csv', '.jpg', '.jpeg', '.png', '.dcm', '.pt', '.pth', '.onnx', '.engine', '.zip', '.7z', '.tar', '.gz'}


def forbidden(path):
    parts = Path(path).parts
    return (bool(parts) and parts[0].casefold() in PRIVATE_ROOTS) or Path(path).suffix.casefold() in PRIVATE_SUFFIXES or any(x.startswith('.env') for x in parts)


def main():
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True)
    paths = [p.decode('utf-8') for p in result.stdout.split(b'\0') if p]
    bad = [p for p in paths if forbidden(p)]
    if bad:
        print('REFUSED: private data/artifacts are in Git index:', file=sys.stderr)
        for path in bad:
            print(path, file=sys.stderr)
        raise SystemExit(1)
    if not paths:
        print('Git index is empty; nothing staged. No publication performed.')
    else:
        print(f'PASS: {len(paths)} indexed paths; no recognized private data/model files. Review source for secrets manually.')


if __name__ == '__main__':
    main()
