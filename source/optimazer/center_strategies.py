from scipy.optimize import minimize

from abc import ABC, abstractmethod
import numpy as np
class CenterStrategy(ABC):
    # zakładamy że vectors są posortowani po fitness function
    @abstractmethod
    def compute_center(self, vectors: np.ndarray) -> np.ndarray:
        pass

class ArithmeticMeanCenterStrategy(CenterStrategy):

    def compute_center(self, vectors:np.ndarray) -> np.ndarray:
        return np.mean(vectors, axis=0)

class MedianCenterStrategy(CenterStrategy):

    def compute_center(self, vectors: np.ndarray) -> np.ndarray:
        return np.median(vectors, axis=0)

class WeightedMeanCenterStrategy(CenterStrategy):
    # weights w kolejności ideksów zgodnymi z vectors
    def compute_center(self, weights, vectors: np.ndarray) -> np.ndarray:
        return np.average(vectors, axis=0, weights = weights)

class TrimmedMeanCenterStrategy(CenterStrategy):

    def compute_center(self, vectors, trimmed_proportion = 0.1) -> np.ndarray:
        n = len(vectors)
        k = int(n * trimmed_proportion)
        # sortujemy po sumie wartości w każdym wektorze
        sorted_vectors = vectors[np.argsort(np.sum(vectors, axis=1))]
        trimmed_vectors = sorted_vectors[k : n-k]
        return np.mean(trimmed_vectors, axis = 0)
