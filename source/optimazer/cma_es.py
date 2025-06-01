import source.optimazer.center_strategies as mn

class cma_es:
    def __init__(self, x0, sigma = 0.5, mean_strategy = mn.ArithmeticMeanStrategy, seed = 44):
        self.x0 = x0
        self.sigma = sigma
        self.mean_strategy = mean_strategy
        self.seed = seed
    
    def optimize(self, func):
        pass


        