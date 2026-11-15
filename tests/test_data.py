import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train
from audit_data import audit


class DataTests(unittest.TestCase):
    def test_validation_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(42)
            for split in ("Training", "Testing"):
                for label in train.CLASSES:
                    folder = root / split / label
                    folder.mkdir(parents=True)
                    for i in range(10):
                        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / f"{i}.png")
            with patch.object(train, "DATA", root):
                training, validation, testing = train.get_loaders(4)
                _, repeated, _ = train.get_loaders(4)
            self.assertEqual(len(validation.dataset), 4)
            self.assertEqual(len(testing.dataset), 40)
            self.assertEqual(validation.dataset.indices, repeated.dataset.indices)
            self.assertFalse(set(training.dataset.indices) & set(validation.dataset.indices))
            self.assertEqual(set(training.dataset.indices) | set(validation.dataset.indices), set(range(40)))
            self.assertIsNot(training.dataset.dataset, validation.dataset.dataset)
            self.assertTrue(torch.equal(validation.dataset[0][0], validation.dataset[0][0]))
            self.assertTrue(any(isinstance(t, train.transforms.RandomRotation)
                                for t in training.dataset.dataset.transform.transforms))

    def test_audit_detects_reencoded_pixels_across_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / split for split in ("Training", "Testing")]
            for root in roots:
                (root / "glioma").mkdir(parents=True)
            image = Image.new("RGB", (12, 12), "red")
            image.save(roots[0] / "glioma" / "a.png")
            image.save(roots[1] / "glioma" / "b.bmp")
            report = audit(roots)
            self.assertEqual(len(report["cross_root_groups"]), 1)
            Image.new("RGB", (12, 12), "blue").save(roots[1] / "glioma" / "b.bmp")
            self.assertEqual(audit(roots)["duplicate_groups"], [])


if __name__ == "__main__":
    unittest.main()
