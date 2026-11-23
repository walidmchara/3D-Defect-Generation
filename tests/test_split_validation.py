import unittest

from src.split_validation import validate_disjoint_samples


class SyntheticSplitTests(unittest.TestCase):
    def test_distinct_samples_pass(self):
        validate_disjoint_samples({"train": [b"one"], "validation": [b"two"], "test": [b"three"]})

    def test_identical_clean_sample_across_splits_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Identical clean volumes"):
            validate_disjoint_samples({"train": [b"one"], "validation": [b"two"], "test": [b"one"]})

    def test_duplicate_within_same_split_does_not_cross_split_boundary(self):
        validate_disjoint_samples({"train": [b"one", b"one"], "test": [b"two"]})


if __name__ == "__main__":
    unittest.main()
