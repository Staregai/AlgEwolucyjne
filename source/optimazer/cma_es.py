import optimazer.center_strategies as mn
import numpy as np
from cma_parameters import CMAParameters 

class cma_es:

    def __init__(self, cma_parameters: CMAParameters):

        self.set_parameters(parameters = cma_parameters)
        # aktualny punkt środkowy
        self.m = np.copy(self.x0)
        # macierz kowariancji
        self.C = np.eye(self.dim)
        # sciezka ewolucji dla sigma
        self.p_sigma = np.zeros(self.dim)
        # sciezka ewolucji dla macierzy kowariancji
        self.p_c = np.zeros(self.dim) 
        # oczekiwana dlugosc wektora rozkładu normalnego
        self.E_norm = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        # ustawienie seedu
        np.random.seed(self.seed)

    def set_parameters(self, parameters: CMAParameters):
        self.x0 = parameters.x0
        self.dim = len(self.x0)
        self.sigma = parameters.sigma
        self.max_iter = parameters.max_iter
        self.center_strategy = parameters.center_strategy
        self.seed = parameters.seed
        self.pop_size = parameters.pop_size
        self.mu = parameters.mu
        self.c_sigma = parameters.c_sigma
        self.d_sigma = parameters.d_sigma
        self.c_c = parameters.c_c
        self.c_1 = parameters.c_1
        self.c_mu = parameters.c_mu

    def optimize(self, func):

        for iter in range(self.max_iter):
            pop, d_list = self.generate_pop()

            fitness = np.array([func(x) for x in pop])
            idx = np.argsort(fitness)
            best_idx = idx[:self.mu]
            best_pop = pop[best_idx]
            best_d = d_list[best_idx]
            best_fitness = fitness[best_idx]

            # Update m
            delta = self.compute_new_center(best_d, best_fitness)
            self.m += self.sigma * delta

            # Aktualizacja ścieżek i parametrów
            self.update_path_sigma(delta)
            self.update_sigma()
            self.update_path_c(delta)
            self.update_covariance(best_d)

        return self.m

    def compute_new_center(self, center_inputs, weights = None):
        if isinstance(self.center_strategy, mn.WeightedFitnessCenterStrategy):
            center = self.center_strategy.compute_center(weights, center_inputs)
        else:
            center = self.center_strategy.compute_center(center_inputs)
        return center

    def generate_pop(self):
        d_list = []
        pop = []
        for _ in range(self.pop_size):
            d = np.random.multivariate_normal(np.zeros(self.dim), self.C)
            x = self.m + self.sigma * d
            d_list.append(d)
            pop.append(x)
        return np.array(pop), np.array(d_list)

    def update_path_sigma(self, delta):
        D, V = np.linalg.eigh(self.C)
        C_inv_sqrt = V @ np.diag(1 / np.sqrt(D)) @ V.T
        # C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C)).T
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu) * (C_inv_sqrt @ delta)

    def update_sigma(self):
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        self.sigma = self.sigma * np.exp((self.c_sigma / self.d_sigma) * (norm_p_sigma / self.E_norm - 1))
        self.sigma = np.clip(self.sigma, 1e-8, 1e2)

    def update_path_c(self, delta):
        self.p_c = (1.0 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2.0 - self.c_c) * float(self.mu)) * delta

    def update_covariance(self, best_d):
        rank_one = self.c_1 * np.outer(self.p_c, self.p_c)
        rank_mu = self.c_mu * np.mean([np.outer(d, d) for d in best_d], axis=0)
        self.C = (1 - self.c_1 - self.c_mu) * self.C + rank_one + rank_mu
