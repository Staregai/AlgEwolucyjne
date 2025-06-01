import unittest
import numpy as np
from optimazer.center_strategies import (
    ArithmeticMeanCenterStrategy,
    MedianCenterStrategy,
    WeightedMeanCenterStrategy,
    TrimmedMeanCenterStrategy,
)

# python -m unittest tests.mean_starategies_tests
class TestCenterStrategy(unittest.TestCase):
    def test_arithmetic_Center(self):
        vectors = np.array([[1, 2], [3, 4]])
        strategy = ArithmeticMeanCenterStrategy()
        result = strategy.compute_center(vectors)
        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_median_center(self):
        vectors = np.array([[1, 5], [3, 3], [5, 1]])
        strategy = MedianCenterStrategy()
        result = strategy.compute_center(vectors)
        expected = np.array([3.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_trimmed_mean_center(self):
        vectors = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [100, 100]  # outlier
        ])
        strategy = TrimmedMeanCenterStrategy()
        result = strategy.compute_center(vectors, trimmed_proportion=0.25)
        expected = np.mean(np.array([[2, 2], [3, 3]]), axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_weighted_mean_center(self):
        vectors = np.array([
            [1, 1],
            [3, 3]
        ])
        weights = np.array([3, 1])
        strategy = WeightedMeanCenterStrategy(weights=weights)
        result = strategy.compute_center(vectors)
        expected = [1.5, 1.5]
        np.testing.assert_array_almost_equal(result, expected)

    

if __name__ == "__main__":
    unittest.main()
