import unittest
from pathlib import Path
from cxr.safety import safe_output


class SafetyTests(unittest.TestCase):
    def test_raw_output_rejected(self):
        raw = Path('cxr_first').resolve()
        for path in (raw, raw / 'test.csv', raw / 'train' / 'a.jpg', raw / 'x' / '..' / 'train.csv'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_output(path, raw)

    def test_input_collision_rejected(self):
        checkpoint = Path('outputs/run/best.pt')
        with self.assertRaises(ValueError):
            safe_output(checkpoint, inputs=[checkpoint])

    def test_output_allowed(self):
        self.assertEqual(safe_output('outputs/result.csv'), Path('outputs/result.csv').resolve())


if __name__ == '__main__':
    unittest.main()
