"""Check complete submission coverage without torch or decoding any image."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cxr.data import _read_metadata
from cxr.submission import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    records = _read_metadata(args.metadata)
    summary = validate_submission(args.submission, records)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
