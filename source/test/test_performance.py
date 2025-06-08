import unittest
import numpy as np
from optimizer.cma_es import cma_es
from functions.benchmark import benchmark_functions,benchmark_bounds
import optimizer.center_strategies as mn
from optimizer.cma_parameters import CMAParameters
import time

class TestCMAESPerformance(unittest.TestCase):
    def setUp(self):
        self.dim = 5
        self.x0 = np.zeros(self.dim)
        self.max_iter = 1000
        self.sigma = 0.5

    def run_optimizer(self, func, center_strategy, known_min, tol=1e-2):
        params = CMAParameters.basic_from_literature(
            dim=self.dim, center_strategy=center_strategy, seed=42
        )
        params.max_iter = self.max_iter
        params.sigma = self.sigma
        optimizer = cma_es(x0=self.x0, parameters=params)
        result = optimizer.optimize(func)
        return result


    def function_performance(self, func_name):
        strategies = [
            mn.ArithmeticMeanCenterStrategy(),
            mn.MedianCenterStrategy(),
            mn.WeightedFitnessCenterStrategy(),
            mn.WeightedRankCenterStrategy(),
            mn.TrimmedMeanCenterStrategy(),
        ]
        tol = 5e-2
        max_time = 5.0  # sekundy
        for strategy in strategies:
            start = time.time()
            result = self.run_optimizer(benchmark_functions.get(func_name, None), strategy, known_min=1.0, tol=tol)
            elapsed = time.time() - start
            fx = benchmark_functions.get(func_name, None)(result)
            print(f"{func_name} {type(strategy).__name__}: f(x)={fx:.3e}, czas={elapsed:.2f}s")
            self.assertTrue(
                fx < tol and elapsed < max_time,
                msg=f"{type(strategy).__name__} failed: f(x)={fx}, czas={elapsed}s"
            )

    def test_all(self):
        for func_name in benchmark_functions.keys():
            with self.subTest(func=func_name):
                self.function_performance(func_name)

if __name__ == "__main__":
    unittest.main()