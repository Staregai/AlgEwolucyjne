import numpy as np
from  optimizer.cma_parameters import * 
class CMAESLogger:
    def __init__(self, cma_parameters, log_step=10):
        self.log_step = log_step
        self.cma_parameters: CMAParameters = cma_parameters
        self.population_history = []
        self.m_history = []
        self.sigma_history = []
        self.best_fitness_history = []

    def log(self, iteration, population, m, sigma, best_fitness):
        if (
            iteration == 0 or 
            iteration % self.log_step == 0
            or iteration == self.cma_parameters.max_iter - 1
        ):
            self.population_history.append(population)
            self.m_history.append(np.copy(m))
            self.sigma_history.append(sigma)
            self.best_fitness_history.append(best_fitness)
    