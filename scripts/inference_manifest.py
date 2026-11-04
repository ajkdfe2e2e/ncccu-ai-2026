"""Inventory explicitly selected inference files. No archive extraction or upload.
List EVERY file actually loaded at inference, including config, engines and
normalization metadata. This tool cannot infer undeclared runtime dependencies.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cxr.safety import safe_output

LIMIT_BYTES = 500_000_000  # Conservative decimal MB, not 500 MiB.


def inventory(paths):
    entries, seen = [], set()
    for value in paths:
        path = Path(value).resolve(strict=True)
        if path in seen:
            continue
        if not path.is_file():
            raise ValueError(f'Expected an explicit regular file, not a directory: {path}')
        seen.add(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        entries.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    if not entries:
        raise ValueError('At least one inference file required')
    total = sum(entry['bytes'] for entry in entries)
    return dict(files=entries, total_bytes=total, limit_bytes=LIMIT_BYTES,
                within_size_limit=total <= LIMIT_BYTES,
                declared_files_only=True,
                note='User must declare every actually loaded file; not organizer certification.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', type=Path, nargs='+')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'inference_manifest.json')
    a = p.parse_args()
    output = safe_output(a.output, inputs=a.files)
    report = inventory(a.files)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['within_size_limit']:
        raise SystemExit('Inference artifact size exceeds 500 MB')


if __name__ == '__main__':
    main()
