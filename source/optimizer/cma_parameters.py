import numpy as np
from optimizer.center_strategies import *


class CMAParameters:

    def __init__(
        self,
        sigma=0.5,
        max_iter=1000,
        center_strategy: CenterStrategy  = ArithmeticMeanCenterStrategy(),
        seed=44,
        pop_size=None,
        mu=None,
        c_sigma=None,
        d_sigma=None,
        c_c=None,
        c_1=None,
        c_mu=None,
        logging=None,
    ):
        # wariacja
        self.sigma = sigma
        # max iter
        self.max_iter = max_iter
        # strategia oblicznaia punktu środkowego
        self.center_strategy = center_strategy
        # seed generatora
        self.seed = seed
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
        # logowanie
        self.loging = True

    @staticmethod
    def basic(x0, center_strategy=None, seed=44):
        return CMAParameters(
            x0=x0,
            sigma=0.5,
            max_iter=300,
            center_strategy=center_strategy or ArithmeticMeanCenterStrategy(),
            seed= 44 ,
            pop_size=None,
            mu=None,  
            )

    @staticmethod
    def basic_from_literature(dim, center_strategy=None, seed=44):

        pop_size = (4 + int(3 * np.log(dim)))
        mu = pop_size // 2
        c_sigma = 0.3
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(
            1 - c_1,
            2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu),
        )

        return CMAParameters(
            sigma=0.5,
            mu=mu,
            pop_size= pop_size,
            c_sigma=0.3,
            d_sigma=d_sigma,
            c_c=c_c,
            c_1=c_1,
            c_mu=c_mu,
            max_iter = 1000,
            center_strategy=center_strategy or ArithmeticMeanCenterStrategy(),
            seed=seed,
        )
