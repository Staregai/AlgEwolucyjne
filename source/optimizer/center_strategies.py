from scipy.optimize import minimize

from abc import ABC, abstractmethod
import numpy as np


class CenterStrategy(ABC):
    # zakładamy że vectors są posortowani po fitness function
    @abstractmethod
    def compute_center(self, vectors: np.ndarray) -> np.ndarray:
        pass


class ArithmeticMeanCenterStrategy(CenterStrategy):

    def compute_center(self, vectors: np.ndarray) -> np.ndarray:
        return np.mean(vectors, axis=0)


class MedianCenterStrategy(CenterStrategy):

    def compute_center(self, vectors: np.ndarray) -> np.ndarray:
        return np.median(vectors, axis=0)


class WeightedFitnessCenterStrategy(CenterStrategy):
    # weights w kolejności ideksów zgodnymi z vectors
    def compute_center(self, weights, vectors: np.ndarray) -> np.ndarray:
        return np.average(vectors, axis=0, weights=weights)


class WeightedRankCenterStrategy(CenterStrategy):
    # weights w kolejności ideksów zgodnymi z vectors, weights jako rangi
    def compute_center(self, vectors: np.ndarray) -> np.ndarray:
        weights = np.linspace(len(vectors), 1, len(vectors))
        return np.average(vectors, axis=0, weights=weights)


class TrimmedMeanCenterStrategy(CenterStrategy):

    def compute_center(self, vectors: np.ndarray, trimmed_proportion=0.1) -> np.ndarray:

        n = len(vectors)
        k = max(1, int(n * trimmed_proportion))
        trimmed_mean = np.mean(vectors[: n - k], axis=0)
        return trimmed_mean


enter_strategies = [
    ArithmeticMeanCenterStrategy(),
    MedianCenterStrategy(),
    WeightedFitnessCenterStrategy(),
    WeightedRankCenterStrategy(),
    TrimmedMeanCenterStrategy(),
]
