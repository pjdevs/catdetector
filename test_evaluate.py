from argparse import Namespace
import unittest

from evaluate import resolve_thresholds


class ResolveThresholdsTest(unittest.TestCase):
    def test_uses_global_threshold_by_default(self):
        args = Namespace(threshold=0.5, vickie_threshold=None, oka_threshold=None)

        self.assertEqual(resolve_thresholds(args), (0.5, 0.5))

    def test_allows_per_cat_threshold_overrides(self):
        args = Namespace(threshold=0.5, vickie_threshold=0.45, oka_threshold=0.44)

        self.assertEqual(resolve_thresholds(args), (0.45, 0.44))

    def test_allows_single_cat_threshold_override(self):
        args = Namespace(threshold=0.5, vickie_threshold=None, oka_threshold=0.44)

        self.assertEqual(resolve_thresholds(args), (0.5, 0.44))


if __name__ == "__main__":
    unittest.main()
