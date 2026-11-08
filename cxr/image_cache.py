"""Optional lossless cache for deterministic RGB letterbox preprocessing."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image, __version__ as PILLOW_VERSION

from .engine import ImageDataset, Letterbox, image_transform


# Bump when deterministic RGB conversion or Letterbox geometry changes.
PREPROCESS_VERSION = 1


class CachedImageDataset(ImageDataset):
    """ImageDataset that caches only RGB conversion and deterministic letterboxing."""

    def __init__(self, records, size=224, training=False, cache_root=None):
        if cache_root is None:
            raise ValueError('cache_root is required')
        self.records = records
        self.size = size
        self.training = training
        self.cache_root = Path(cache_root).resolve()
        self.transform = image_transform(size, training)
        self.letterbox = Letterbox(size)
        self._validate_cache_location()

    def _validate_cache_location(self):
        for record in self.records:
            source = Path(record['path']).resolve()
            image_dir = source.parent
            if self.cache_root == image_dir or self.cache_root.is_relative_to(image_dir):
                raise ValueError(f'cache_root must be outside image directory tree: {image_dir}')

    def _cache_path(self, source):
        source = Path(source).resolve()
        stat = source.stat()
        key = json.dumps({
            'path': str(source),
            'size_bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
            'image_size': self.size,
            'pillow': PILLOW_VERSION,
            'preprocess_version': PREPROCESS_VERSION,
        }, sort_keys=True, separators=(',', ':')).encode('utf-8')
        digest = hashlib.sha256(key).hexdigest()
        return self.cache_root / digest[:2] / f'{digest}.png'

    def _read_cached(self, path):
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != 'RGB' or image.size != (self.size, self.size):
                    raise ValueError(f'Invalid cached image mode/size at {path}')
                return image.copy()
        except (OSError, SyntaxError) as exc:
            raise ValueError(f'Corrupt cached PNG at {path}') from exc

    def _write_cached(self, source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=f'.{target.stem}.', suffix='.tmp', dir=target.parent)
            os.close(fd)
            with Image.open(source) as image:
                prepared = self.letterbox(image.convert('RGB'))
            prepared.save(temporary, format='PNG', compress_level=1)
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def __getitem__(self, idx):
        record = self.records[idx]
        cache_path = self._cache_path(record['path'])
        if cache_path.exists():
            image = self._read_cached(cache_path)
        else:
            self._write_cached(record['path'], cache_path)
            image = self._read_cached(cache_path)
        tensor = self.transform(image)
        labels = record['labels']
        target = torch.tensor(labels if labels is not None else [0.] * 10, dtype=torch.float32)
        return tensor, target
