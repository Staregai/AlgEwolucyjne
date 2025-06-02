import unittest
import numpy as np
from optimazer.cma_es import cma_es
from functions.benchmark import *
import optimazer.center_strategies as mn


# python -m unittest tests.cma_es_tests.py
class cma_es_correctness_tests(unittest.TestCase):

    def run_optimizer_and_check(self, func, dim=10, known_min=0.0, tol=1e-2, **kwargs):
        optimizer = cma_es(x0=np.zeros(dim), pop_size=10, **kwargs)
        optimizer.optimize(func, max_iter=1000)
        print(optimizer.m)
        self.assertTrue(
            np.allclose(optimizer.m, known_min, atol=tol),
            msg=f"{func.__name__} failed: got {optimizer.m}, expected {known_min}",
        )

    def test_ackley(self):
        self.run_optimizer_and_check(ackley, dim=10, known_min=0.0, use_direct_mean=True)

    def test_ackley(self):
        self.run_optimizer_and_check(ackley, dim=10, known_min=0.0)

    # def test_rosenbrock(self):
    #     self.run_optimizer_and_check(rosenbrock, dim=2, known_min=1.0, tol=1e-2)




if __name__ == "__main__":
    unittest.main()
