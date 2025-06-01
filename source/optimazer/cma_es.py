import optimazer.center_strategies as mn
import numpy as np

class cma_es:
    def __init__(self, x0, sigma = 0.5, center_strategy: mn.CenterStrategy = mn.ArithmeticMeanCenterStrategy, seed = 44, pop_size = 20, mu = 10, c_sigma = 0.3, d_sigma = 0.7,  c_c = 0.2, c_1 = 0.2, c_mu = 0.2):
        # punkt startowy
        self.x0 = x0
        # wariacja
        self.sigma = sigma
        # strategia oblicznaia punktu środkowego
        self.center_strategy = center_strategy
        # seed generatora
        self.seed = seed
        # rozmiar populacji
        self.pop_size = pop_size   
        # liczba selekcji najepszych rozwiązań
        self.mu = mu 
        # współczynnik ewolucji dla sigma (szybkość adaptacji kroku ewolucji)
        self.c_sigma = c_sigma
        # współczynnik tłumienia ewolucji dla sigma (regulacja tempa ewolucji)
        self.d_sigma = d_sigma
        # współczynnik ewolucji dla ścieżki ewolucji
        self.c_c = c_c
        # współczynnik ewolucji dla macierzy kowariancji
        self.c_1 = c_1
        # współczynnik ewolucji dla średniej macierzy kowariancji
        self.c_mu = c_mu

        np.random.seed(self.seed)

        self.dim = len(self.x0)
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

    def optimize(self, func, max_iter=1000):

        for _ in range(max_iter):
            pop, d_list = self.generate_pop()

            # TODO () WYTESTUJ
            fitness = np.array([func(x) for x in pop])
            idx = np.argsort(fitness)
            best_idx = idx[:self.mu]
            best_d = d_list[best_idx]
            delta = np.mean(best_d, axis=0)

            # Update m
            self. m = self.m + self.sigma * delta

            # Update evolution paths and parameters
            self.update_path_sigma(delta)
            self.update_sigma()
            self.update_path_c(delta)
            self.update_covariance(best_d)

        return self.m

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
        C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.C)).T
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu) * (C_inv_sqrt @ delta)

    def update_sigma(self):
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        self.sigma = self.sigma * np.exp((self.c_sigma / self.d_sigma) * (norm_p_sigma / self.E_norm - 1))

    def update_path_c(self, delta):
        self.p_c = (1.0 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2.0 - self.c_c) * float(self.mu)) * delta

# TODO TEST() 
    def update_covariance(self, best_d):
        rank_one = self.c_1 * np.outer(self.p_c, self.p_c)
        rank_mu = self.c_mu * np.mean([np.outer(d, d) for d in best_d], axis=0)
        self.C = (1 - self.c_1 - self.c_mu) * self.C + rank_one + rank_mu
