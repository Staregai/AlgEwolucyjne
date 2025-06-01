import unittest
import numpy as np
from optimazer.cma_es import cma_es

# python -m unittest tests.cma_es_tests.py
class cma_es_tests(unittest.TestCase):

    def test_generate_population(self):
        pop_size = 1000
        dim = 2
        sigma = 0.5

        optimizer = cma_es(x0=np.zeros(dim), pop_size=1000)
        pop, d_list = optimizer.generate_pop()

        d_mean = np.mean(d_list, axis=0)
        d_std = np.std(d_list, axis=0)

        self.assertTrue(np.all(np.abs(d_mean) < 0.2)) 
        self.assertTrue(np.all(np.abs(d_std - 1) < 0.2))

        p_mean = np.mean(pop, axis=0)
        p_std = np.std(pop, axis=0)

        self.assertTrue(np.all(np.abs(p_mean) < 0.2))
        self.assertTrue(np.all(np.abs(p_std - sigma) < 0.1))

        np.testing.assert_array_equal(pop.shape, (pop_size, dim))
        np.testing.assert_array_equal(d_list.shape, (pop_size, dim))

    def test_generate_population_second(self):
        pop_size = 1000
        dim = 4
        sigma = 0.5

        optimizer = cma_es(x0=np.zeros(dim), pop_size=pop_size, sigma=0.5)
        optimizer.C = np.eye(dim) * 4 
        optimizer.m = [10, 10, 10, 10]

        pop, d_list = optimizer.generate_pop()

        d_mean = np.mean(d_list, axis=0)
        self.assertTrue(np.allclose(d_mean, [0, 0, 0, 0], atol=0.3))

        p_mean = np.mean(pop, axis=0)
        self.assertTrue(np.allclose(p_mean, [10, 10, 10, 10], atol=0.1))

    def test_cma_es_update_path_sigma(self):
        dim = 2
        optimizer = cma_es(x0=np.zeros(dim), pop_size=1000)
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
        optimizer = cma_es(x0=np.zeros(dim), pop_size=1000)
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

        optimizer = cma_es(x0=np.zeros(dim), pop_size=1000)
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


if __name__ == "__main__":
    unittest.main()
