"""Synthetic tests for lossless preprocessing cache behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from cxr.engine import ImageDataset
from cxr.image_cache import CachedImageDataset


class CachedImageDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image_dir = self.root / 'images' / 'subject' / 'study'
        self.image_dir.mkdir(parents=True)
        self.source = self.image_dir / 'sample.png'
        pixels = torch.arange(19 * 13 * 3, dtype=torch.uint8).reshape(13, 19, 3).numpy()
        Image.fromarray(pixels).save(self.source)
        self.records = [{'path': str(self.source), 'labels': [1.] + [0.] * 9}]
        self.cache = self.root / 'cache'

    def test_cached_and_uncached_match_eval_and_seeded_training_bitwise(self):
        for training in (False, True):
            original = ImageDataset(self.records, size=32, training=training)
            cached = CachedImageDataset(self.records, size=32, training=training, cache_root=self.cache)
            torch.manual_seed(101)
            expected = original[0]
            torch.manual_seed(101)
            actual = cached[0]
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertTrue(torch.equal(expected[1], actual[1]))
            torch.manual_seed(202)
            expected_again = original[0][0]
            torch.manual_seed(202)
            actual_again = cached[0][0]
            self.assertTrue(torch.equal(expected_again, actual_again))

    def test_source_change_and_size_change_get_separate_cache_entries(self):
        first = CachedImageDataset(self.records, size=24, cache_root=self.cache)
        first[0]
        first_key = first._cache_path(self.source)
        second = CachedImageDataset(self.records, size=32, cache_root=self.cache)
        second[0]
        second_key = second._cache_path(self.source)
        self.assertNotEqual(first_key, second_key)
        self.source.write_bytes(self.source.read_bytes() + b'changed')
        self.assertNotEqual(first_key, first._cache_path(self.source))

    def test_failed_replace_leaves_no_target_or_temporary_file(self):
        dataset = CachedImageDataset(self.records, size=24, cache_root=self.cache)
        target = dataset._cache_path(self.source)
        with patch('cxr.image_cache.os.replace', side_effect=OSError('simulated interruption')):
            with self.assertRaisesRegex(OSError, 'interruption'):
                dataset[0]
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.iterdir()), [])

    def test_cache_root_inside_image_directory_tree_is_rejected(self):
        for root in (self.image_dir, self.image_dir / 'cache'):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, 'outside image directory tree'):
                CachedImageDataset(self.records, cache_root=root)

    def test_corrupt_cache_raises_clear_error(self):
        dataset = CachedImageDataset(self.records, size=24, cache_root=self.cache)
        target = dataset._cache_path(self.source)
        target.parent.mkdir(parents=True)
        target.write_bytes(b'not png')
        with self.assertRaisesRegex(ValueError, 'Corrupt cached PNG'):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
