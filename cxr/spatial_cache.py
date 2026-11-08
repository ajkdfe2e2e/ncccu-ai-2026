"""Resumable, fingerprinted spatial tensors for a frozen encoder prefix."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_sha256(values):
    # Blocks are bounded by the committed row interval, not the entire matrix.
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _verify_progress(values, progress):
    completed = progress.get("completed")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= len(values):
        raise ValueError("invalid spatial cache committed cursor")
    cursor = 0
    for block in progress.get("blocks", []):
        end = block.get("end")
        if (block.get("start") != cursor or isinstance(end, bool)
                or not isinstance(end, int) or not cursor < end <= completed):
            raise ValueError("invalid committed spatial cache block")
        if _block_sha256(values[cursor:end]) != block.get("sha256"):
            raise ValueError("committed spatial cache block changed")
        cursor = end
    if cursor != completed:
        raise ValueError("spatial cache cursor has no committed block evidence")


def extract_prefix_cache(prefix, batches, folder, spec, *, device, commit_rows=100,
                         reserve_bytes=1024 ** 3, progress_callback=None):
    """Cache exact dtype prefix outputs; resume only hash-verified committed rows.

    ``batches(start)`` yields contiguous ``(offset, pixel_tensor)`` batches.
    ``spec`` includes ``shape``, ``dtype`` and all external input fingerprints.
    """
    folder = Path(folder)
    shape, dtype = tuple(spec["shape"]), np.dtype(spec["dtype"])
    if (len(shape) != 4 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape)
            or dtype not in (np.dtype("float16"), np.dtype("float32"))
            or commit_rows < 1 or reserve_bytes < 0):
        raise ValueError("invalid spatial cache shape/dtype/commit policy")
    folder.mkdir(parents=True, exist_ok=True)
    input_path, matrix_path = folder / "input.json", folder / "prefix.npy"
    progress_path = folder / "progress.json"
    if input_path.exists():
        if read_json(input_path) != spec:
            raise ValueError("spatial cache input fingerprint mismatch")
    else:
        if matrix_path.exists() or progress_path.exists():
            raise ValueError("unregistered spatial cache files already exist")
        atomic_json(input_path, spec)
    progress = read_json(progress_path) if progress_path.exists() else {"completed": 0, "blocks": []}
    if matrix_path.exists():
        values = np.load(matrix_path, mmap_mode="r+", allow_pickle=False)
        if values.shape != shape or values.dtype != dtype:
            raise ValueError("spatial cache matrix shape/dtype mismatch")
    else:
        if progress["completed"]:
            raise ValueError("spatial cache claims committed rows without a matrix")
        needed = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize + 4096
        if shutil.disk_usage(folder).free < needed + reserve_bytes:
            raise OSError(f"insufficient space for spatial cache: need {needed + reserve_bytes} bytes")
        values = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=dtype, shape=shape)
    _verify_progress(values, progress)
    if progress.get("status") == "complete":
        if progress["completed"] != len(values) or file_sha256(matrix_path) != progress.get("sha256"):
            raise ValueError("completed spatial cache hash mismatch")
        return values
    prefix.eval()
    device = torch.device(device)
    committed = cursor = progress["completed"]
    started, initial = time.monotonic(), cursor
    blocks = list(progress["blocks"])
    with torch.inference_mode():
        for offset, pixels in batches(cursor):
            if offset != cursor or not isinstance(pixels, torch.Tensor) or not len(pixels):
                raise ValueError("prefix batches are not contiguous at the committed cursor")
            end = cursor + len(pixels)
            if end > len(values):
                raise ValueError("prefix batch exceeds registered cache shape")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = prefix(pixels.to(device))
            batch = result.cpu().numpy()
            if batch.shape != (len(pixels), *shape[1:]) or batch.dtype != dtype or not np.isfinite(batch).all():
                raise ValueError("prefix output shape/dtype/finite contract differs from smoke")
            values[cursor:end] = batch
            cursor = end
            if cursor - committed >= commit_rows or cursor == len(values):
                values.flush()
                blocks.append({"start": committed, "end": cursor, "sha256": _block_sha256(values[committed:cursor])})
                progress = {"status": "extracting", "completed": cursor, "total": len(values), "blocks": blocks,
                            "segment_images_per_second": (cursor - initial) / max(time.monotonic() - started, 1e-9)}
                atomic_json(progress_path, progress)
                committed = cursor
                if progress_callback:
                    progress_callback(progress)
    if cursor != len(values):
        raise ValueError("prefix batch iterator ended before all rows were extracted")
    progress.update(status="complete", sha256=file_sha256(matrix_path))
    atomic_json(progress_path, progress)
    if progress_callback:
        progress_callback(progress)
    return values
