import unittest
import numpy as np
from optimizer.cma_es import cma_es
from functions.benchmark import (
    benchmark_functions,
    benchmark_bounds,
    benchmark_known_min,
)
import optimizer.center_strategies as mn
from optimizer.cma_parameters import CMAParameters
import time

import csv
import os
import pickle
import math

RESULTS_CSV = "results/performance_results.csv"
RESULTS_PKL = "results/performance_results.pkl"
RESULT_DIR = "results"
CONVERGENCE_DIR = "results/convergence"

SEED_FILE = "experiments/seeds.txt"


# python -m experiments.test_performance
class TestCMAESPerformanceConvergence(unittest.TestCase):
    def setUp(self):
        self.dims = [2, 5, 10, 20]
        self.max_iter = 1000
        self.sigma = 0.5

        with open(SEED_FILE) as f:
            seeds = [int(line.strip()) for line in f]

        self.seeds = seeds

        self.results = {}
        os.makedirs(RESULT_DIR, exist_ok=True)
        os.makedirs(CONVERGENCE_DIR, exist_ok=True)

    def run_optimizer(self, func, params: CMAParameters, x0):

        optimizer = cma_es(x0=x0, parameters=params)
        result = optimizer.optimize(func)

        curve = getattr(optimizer, "f_history", None)
        if curve is None and hasattr(optimizer, "logger"):
            curve = getattr(optimizer.logger, "best_fitness_history", None)
        if curve is not None:
            curve_filename = f"{CONVERGENCE_DIR}/{func.__name__}_{type(params.center_strategy).__name__}_dim{params.dim}_seed{params.seed}.csv"
            with open(curve_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "fx"])
                for i, fx_iter in enumerate(curve):
                    writer.writerow([i * optimizer.logger.log_step, fx_iter])
        return result

    def save_results_to_csv(self, filename=RESULTS_CSV):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["function", "strategy", "dim", "seed", "fx"])
            for (func_name, strategy_name, dim), fx_list in self.results.items():
                for i, fx in enumerate(fx_list):
                    writer.writerow([func_name, strategy_name, dim, self.seeds[i], fx])

    def save_results_pickle(self, filename=RESULTS_PKL):
        with open(filename, "wb") as f:
            pickle.dump(self.results, f)

    def load_results_pickle(self, filename=RESULTS_PKL):
        with open(filename, "rb") as f:
            self.results = pickle.load(f)

    def test_performance_and_collect(self):
        strategies = [
            mn.ArithmeticMeanCenterStrategy(),
            mn.MedianCenterStrategy(),
            mn.WeightedFitnessCenterStrategy(),
            mn.WeightedRankCenterStrategy(),
            mn.TrimmedMeanCenterStrategy(),
        ]
        tol = 5e-2
        max_time = 10.0 
        errors = [] 

        for func_name, func in benchmark_functions.items():
            for dim in self.dims:
                if (func_name in ["michalewicz", "booth"] and dim > 2):
                    continue
                params = CMAParameters.basic_from_literature(dim=dim)
                params.max_iter = int(self.max_iter * math.log(dim))
                (lower, upper) = benchmark_bounds.get(func.__name__, None)
                params.sigma = (upper - lower) / 4
                known_min = benchmark_known_min.get(func.__name__, 0.0)
                x0 = np.clip(
                    known_min + np.random.uniform(lower / 2, upper / 2, size=dim),
                    lower,
                    upper,
                )

                for strategy in strategies:
                    fx_list = []
                    params.center_strategy = strategy
                    for seed in self.seeds:
                        params.seed = seed
                        start = time.time()
                        result = self.run_optimizer(func, params, x0)
                        elapsed = time.time() - start
                        fx = func(result)
                        fx_list.append(fx)
                        print(
                            f"{func_name} dim={dim} {type(strategy).__name__}: f(x)={fx:.3e}, czas={elapsed:.2f}s"
                        )
                        if not (fx < tol and elapsed < max_time):
                            errors.append(
                                f"{func_name} {type(strategy).__name__} dim ={dim} seed={seed}: f(x)={fx}, czas={elapsed}s"
                            )
                    self.results[(func_name, type(strategy).__name__, dim)] = fx_list

        self.save_results_to_csv()
        self.save_results_pickle()

        if errors:
            print("\nNiespełnione przypadki:")
            for err in errors:
                print(err)


if __name__ == "__main__":
    unittest.main()
