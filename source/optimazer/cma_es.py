import source.optimazer.center_strategies as mn
import numpy as np

class cma_es:
    def __init__(self, x0, sigma = 0.5, center_strategy: mn.CenterStrategy = mn.ArithmeticMeanStrategy , seed = 44, pop_size = 20, mu = 10 ):
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

    def optimize(self, func):

        while True: 

            self.generate_pop()
            # sorting()
            # compute delta
            delta = self.center_strategy.compute_center()

            # przesun srodek o delta
            # update center
            # new_center = old_center + sigma * delta

                        # Δ(t)
            self.update_path_sigma()

            self.update_sigma()

            self.update_path_c()

            self.update_covariance()

            # warunek stopu

        pass

    def generate_pop(self) -> np.ndarray:
        # for i = 1 do pop_size
        # d_i ~ N(0, C)
        # x_i = m + sigma * d_i
        # return population
        return []

    def update_center(self):
        return []

    def update_path_sigma(self):
        # aktualizuje p_sigma i p_c albo osobno to rozbić  pewnie lepiej 
        return []
    
    def update_path_c(self):
        # aktualizuje p_sigma i p_c albo osobno to rozbić  pewnie lepiej 
        return []
    
    def update_sigma(self):
        return []

    def update_covariance(self):
        return []