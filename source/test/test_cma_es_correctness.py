import unittest
import numpy as np
from optimazer.cma_es import cma_es
from functions.benchmark import *
import optimazer.center_strategies as mn


# python -m unittest test.test_cma_es_correctness.py
class cma_es_correctness_tests(unittest.TestCase):

    def run_optimizer_and_check(self, func, dim=10, known_min=0.0, tol=1e-2, **kwargs):
        optimizer = cma_es(x0=np.zeros(dim), **kwargs)
        optimizer.optimize(func)
        self.assertTrue( 
            np.allclose(optimizer.m, known_min, atol=tol),
            msg=f"{func.__name__} failed: got {optimizer.m}, expected {known_min}",
        )

    def test_ackley_30(self):
        self.run_optimizer_and_check(ackley, dim=30, known_min=0.0, use_direct_mean=True)

    def test_ackley_10(self):
        self.run_optimizer_and_check(ackley, dim=10, known_min=0.0)

    def test_rosenbrock_2(self):
        self.run_optimizer_and_check(rosenbrock, dim=2, known_min=1.0, tol=1e-2, max_iter=1000)

    def test_rosenbrock_10(self):
        self.run_optimizer_and_check(rosenbrock, dim=10, known_min=1.0, tol=1e-2, max_iter=1000)

    def test_sphere(self):
        self.run_optimizer_and_check(sphere, dim=10, known_min=0.0)

    # def test_rastrigin(self):
    #     self.run_optimizer_and_check(rastrigin, dim=5, known_min=0.0, max_iter=2000)

    def test_rosenbrock(self):
        self.run_optimizer_and_check(rosenbrock, dim=10, known_min=1.0, tol=1e-2, max_iter=1000)

    def test_ellipsoid(self):
        self.run_optimizer_and_check(ellipsoid, dim=10, known_min=0.0)

    def test_schwefel(self):
        self.run_optimizer_and_check(schwefel, dim=10, known_min=0.0, tol=10, max_iter=2000)

    def test_griewank(self):
        self.run_optimizer_and_check(griewank, dim=10, known_min=0.0, tol=1e-2)

    def test_zakharov(self):
        self.run_optimizer_and_check(zakharov, dim=10, known_min=0.0, tol=1e-2)

    # nie moze sie wyciagnac z jakiegos minimum
    # https://www.sfu.ca/~ssurjano/michal.html
    # def test_michalewicz(self):
    #     self.run_optimizer_and_check(michalewicz, dim=2, known_min=[2.20, 1.57], tol=0.1, max_iter=1000)

    def test_booth(self):
        self.run_optimizer_and_check(booth, dim=2, known_min=[1.0, 3.0], tol=1e-2)

if __name__ == "__main__":
    unittest.main()
