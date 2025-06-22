import unittest
import numpy as np
from optimizer.cma_es import cma_es
from functions.benchmark import ackley
import optimizer.center_strategies as mn
from optimizer.cma_parameters import *


# python -m unittest test.testscma_es_.py
class cma_es_unit_tests(unittest.TestCase):

    def test_generate_population(self):
        pop_size = 1000
        dim = 2

        optimizer = cma_es(x0=np.zeros(dim))
        optimizer.pop_size = 1000
        pop, d_list = optimizer.generate_pop()

        d_mean = np.mean(d_list, axis=0)
        d_std = np.std(d_list, axis=0)

        self.assertTrue(np.all(np.abs(d_mean) < 0.2))
        self.assertTrue(np.all(np.abs(d_std - 1) < 0.2))

        p_mean = np.mean(pop, axis=0)
        p_std = np.std(pop, axis=0)

        self.assertTrue(np.all(np.abs(p_mean) < 0.2))
        self.assertTrue(np.all(np.abs(p_std - optimizer.sigma) < 0.1))

        np.testing.assert_array_equal(pop.shape, (pop_size, dim))
        np.testing.assert_array_equal(d_list.shape, (pop_size, dim))

    def test_generate_population_second(self):
        dim = 4
        optimizer = cma_es(x0=np.zeros(dim))
        optimizer.C = np.eye(dim) * 4
        optimizer.m = [10, 10, 10, 10]
        optimizer.pop_size = 1000

        pop, d_list = optimizer.generate_pop()
        d_mean = np.mean(d_list, axis=0)
        self.assertTrue(np.allclose(d_mean, [0, 0, 0, 0], atol=0.3))

        p_mean = np.mean(pop, axis=0)
        self.assertTrue(np.allclose(p_mean, [10, 10, 10, 10], atol=0.1))

    def test_cma_es_update_path_sigma(self):
        dim = 2
        optimizer = cma_es(x0=np.zeros(dim))
        optimizer.C = np.eye(dim)
        optimizer.p_sigma = np.zeros(dim)
        optimizer.c_sigma = 0.5
        optimizer.mu = 10

        delta = np.array([1.0, 2.0])
        optimizer.update_path_sigma(delta)
        # p_sigma = (1 - 0.5) * 0 + sqrt(0.5 * (2-0.5) * 10) * delta
        #         = sqrt(7.5) * delta ≈  2.7386 * [1.0, 2.0]

        expected = np.sqrt(7.5) * delta
        np.testing.assert_array_almost_equal(optimizer.p_sigma, expected, decimal=6)

    def test_cma_es_update_sigma(self):
        dim = 2
        optimizer = cma_es(x0=np.zeros(dim))
        optimizer.sigma = 1.0
        optimizer.p_sigma = np.array([3.0, 4.0])
        optimizer.c_sigma = 0.3
        optimizer.d_sigma = 0.7

        norm_p_sigma = np.linalg.norm(optimizer.p_sigma)

        delta = np.array([1.0, 2.0])
        optimizer.update_sigma()
        sigma = 1 * np.exp((0.3 / 0.7) * ((5 / optimizer.E_norm) - 1))

        np.testing.assert_array_almost_equal(optimizer.sigma, sigma, decimal=6)

    def test_cma_es_update_path_c(self):
        dim = 2

        optimizer = cma_es(x0=np.zeros(dim))
        p_c = np.array([2.0, 2.0])
        optimizer.p_c = p_c
        optimizer.c_c = 0.5
        optimizer.mu = 10
        delta = np.array([1.0, 2.0])
        optimizer.update_path_c(delta)

        expected = (
            p_c * (1.0 - optimizer.c_c)
            + np.sqrt(optimizer.c_c * (2.0 - optimizer.c_c) * 10.0) * delta
        )

        np.testing.assert_array_almost_equal(optimizer.p_c, expected, decimal=6)

    def test_update_covariance(self):
        dim = 2
        optimizer = cma_es(x0=np.zeros(dim))

        optimizer.C = np.eye(dim)

        optimizer.c_1 = 0.2
        optimizer.c_mu = 0.3

        optimizer.p_c = np.array([1.0, 2.0])
        rank_one = optimizer.c_1 * np.outer(optimizer.p_c, optimizer.p_c)

        d1 = np.array([1.0, 0.0])
        d2 = np.array([0.0, 2.0])
        best_d = np.array([d1, d2])

        rank_mu = optimizer.c_mu * np.mean([np.outer(d, d) for d in best_d], axis=0)

        expected_C = (
            (1 - optimizer.c_1 - optimizer.c_mu) * optimizer.C + rank_one + rank_mu
        )

        optimizer.update_covariance(best_d)

        np.testing.assert_array_almost_equal(optimizer.C, expected_C, decimal=6)

    def test_arithmetic_mean_center(self):
        pop = np.array([[1, 2], [3, 4], [5, 6]])
        cma = cma_es(x0=[0, 0])
        center = cma.compute_new_center(pop)
        assert np.allclose(center, np.mean(pop, axis=0))

    def test_weighted_mean_center(self):
        pop = np.array([[1, 2], [3, 4], [5, 6]])
        weights = np.linspace(len(pop), 1, len(pop))

        parameters = CMAParameters.basic_from_literature(
            dim=2, center_strategy=mn.WeightedFitnessCenterStrategy()
        )
        cma = cma_es(x0=[0, 0], parameters=parameters)
        center = cma.compute_new_center(pop, weights=weights)
        self.assertTrue(np.allclose(center, np.average(pop, axis=0, weights=weights)))

    def test_median_center(self):
        pop = np.array([[1, 2], [3, 4], [5, 100]])
        parameters = CMAParameters.basic_from_literature(
            dim=2, center_strategy=mn.MedianCenterStrategy()
        )
        cma = cma_es(x0=[0, 0], parameters=parameters)
        center = cma.compute_new_center(pop)
        self.assertTrue(np.allclose(center, np.median(pop, axis=0)))

    def test_trimmed_mean_center(self):
        pop = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
                [13, 14],
                [15, 16],
                [17, 18],
                [100, 5],
            ]
        )

        cma = cma_es(
            x0=[0, 0],
            parameters=CMAParameters.basic_from_literature(
                dim=2, center_strategy=mn.TrimmedMeanCenterStrategy()
            ),
        )
        center = cma.compute_new_center(pop)
        # sortujemy po sumie wartości w każdym wektorze
        sorted_vectors = pop[np.argsort(np.sum(pop, axis=1))]
        trimmed_vectors = sorted_vectors[1:-1]  # po jednym z każdej strony (k=1)
        expected = np.mean(trimmed_vectors, axis=0)
        self.assertTrue(np.allclose(center, expected))


if __name__ == "__main__":
    unittest.main()
