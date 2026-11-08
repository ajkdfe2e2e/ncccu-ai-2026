import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cxr.spatial_cache import extract_prefix_cache


class Prefix(torch.nn.Module):
    def forward(self, x):
        return x * 2


def specification():
    return {"shape": [5, 2, 2, 2], "dtype": "float32", "input_sha256": "fixed"}


def batches(start):
    for row in range(start, 5, 2):
        end = min(row + 2, 5)
        yield row, torch.arange(row * 8, end * 8, dtype=torch.float32).reshape(-1, 2, 2, 2)


def run(path, source=batches, spec=None, **kwargs):
    return extract_prefix_cache(Prefix(), source, path, spec or specification(), device="cpu", commit_rows=2, reserve_bytes=0, **kwargs)


def exact_cache_and_completed_replay(tmp_path):
    actual = run(tmp_path)
    expected = (torch.arange(40).reshape(5, 2, 2, 2) * 2).numpy()
    assert np.array_equal(actual, expected)
    def never_called(start):
        raise AssertionError("completed cache re-extracted")
    assert np.array_equal(run(tmp_path, never_called), expected)


def resume_only_committed_rows(tmp_path):
    def interrupted(start):
        yield from list(batches(start))[:1]
        raise RuntimeError("interruption")
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "interruption"):
        run(tmp_path, interrupted)
    seen = []
    def resumed(start):
        seen.append(start)
        yield from batches(start)
    actual = run(tmp_path, resumed)
    assert seen == [2]
    assert np.array_equal(actual.ravel(), np.arange(40) * 2)


def tampered_committed_block_rejected(tmp_path):
    run(tmp_path)
    values = np.load(tmp_path / "prefix.npy", mmap_mode="r+")
    values[0, 0, 0, 0] += 1
    values.flush()
    with unittest.TestCase().assertRaisesRegex(ValueError, "block changed"):
        run(tmp_path)


def changed_spec_and_dtype_rejected(tmp_path):
    run(tmp_path)
    changed = specification() | {"input_sha256": "other"}
    with unittest.TestCase().assertRaisesRegex(ValueError, "fingerprint mismatch"):
        run(tmp_path, spec=changed)
    with unittest.TestCase().assertRaisesRegex(ValueError, "dtype/finite"):
        run(tmp_path / "dtype", spec=specification() | {"dtype": "float16"})


def missing_cursor_evidence_and_discontinuous_batches_rejected(tmp_path):
    run(tmp_path)
    path = tmp_path / "progress.json"
    progress = json.loads(path.read_text())
    progress["blocks"] = []
    path.write_text(json.dumps(progress))
    with unittest.TestCase().assertRaisesRegex(ValueError, "block evidence"):
        run(tmp_path)
    def gap(start):
        yield 1, torch.zeros((2, 2, 2, 2))
    with unittest.TestCase().assertRaisesRegex(ValueError, "not contiguous"):
        run(tmp_path / "gap", gap)


class SpatialCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_exact_cache_and_completed_replay(self):
        exact_cache_and_completed_replay(self.path)

    def test_resume_only_committed_rows(self):
        resume_only_committed_rows(self.path)

    def test_tampered_committed_block_rejected(self):
        tampered_committed_block_rejected(self.path)

    def test_changed_spec_and_dtype_rejected(self):
        changed_spec_and_dtype_rejected(self.path)

    def test_missing_cursor_evidence_and_discontinuous_batches_rejected(self):
        missing_cursor_evidence_and_discontinuous_batches_rejected(self.path)
