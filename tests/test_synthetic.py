import random
import unittest

import numpy as np

from src.dataset import VolumePairDataset
from src.synthetic import generate_sample, make_base_volume


class SyntheticVolumeTests(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)

    def test_base_volume_has_expected_shape_and_range(self):
        volume = make_base_volume(16)

        self.assertEqual(volume.shape, (16, 16, 16))
        self.assertEqual(volume.dtype, np.float32)
        self.assertGreaterEqual(float(volume.min()), 0.0)
        self.assertLessEqual(float(volume.max()), 1.0)

    def test_zero_defect_probability_returns_empty_mask(self):
        clean, defective, mask, label = generate_sample(
            size=16,
            defect_types=["sphere"],
            min_radius=2,
            max_radius=3,
            intensity_min=-0.8,
            intensity_max=-0.3,
            defect_probability=0.0,
        )

        self.assertEqual(label, "none")
        self.assertEqual(clean.shape, (16, 16, 16))
        np.testing.assert_array_equal(defective, clean)
        self.assertEqual(float(mask.sum()), 0.0)

    def test_each_supported_defect_generates_a_nonempty_mask(self):
        for defect_type in ("sphere", "ellipsoid", "crack"):
            with self.subTest(defect_type=defect_type):
                random.seed(7)
                np.random.seed(7)
                _, defective, mask, label = generate_sample(
                    size=16,
                    defect_types=[defect_type],
                    min_radius=2,
                    max_radius=2,
                    intensity_min=-0.8,
                    intensity_max=-0.3,
                    defect_probability=1.0,
                )

                self.assertEqual(label, defect_type)
                self.assertEqual(defective.shape, (16, 16, 16))
                self.assertGreater(float(mask.sum()), 0.0)
                self.assertTrue(np.isfinite(defective).all())
                self.assertGreaterEqual(float(defective.min()), 0.0)
                self.assertLessEqual(float(defective.max()), 1.0)

    def test_unknown_defect_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown defect type"):
            generate_sample(
                size=16,
                defect_types=["unknown"],
                min_radius=2,
                max_radius=2,
                intensity_min=-0.8,
                intensity_max=-0.3,
                defect_probability=1.0,
            )

    def test_dataset_adds_channel_dimension(self):
        clean = np.zeros((2, 8, 8, 8), dtype=np.float32)
        dataset = VolumePairDataset(clean, clean, clean)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(dataset[0][0].shape), (1, 8, 8, 8))
        self.assertEqual(tuple(dataset[0][1].shape), (1, 8, 8, 8))
        self.assertEqual(tuple(dataset[0][2].shape), (1, 8, 8, 8))


if __name__ == "__main__":
    unittest.main()
