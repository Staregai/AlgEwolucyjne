import unittest
import numpy as np
from optimizer.cma_es import cma_es
from functions.benchmark import benchmark_functions, benchmark_bounds
import optimizer.center_strategies as mn
from optimizer.cma_parameters import CMAParameters
import time
from scipy.stats import wilcoxon
import csv
import os
import pickle

RESULTS_CSV = "performance_results.csv"
RESULTS_PKL = "performance_results.pkl"
RESULT_DIR = "results"
CONVERGENCE_DIR = 'results/convergence'

# python -m experiments.test_performance
class TestCMAESPerformance(unittest.TestCase):
    def setUp(self):
        self.dim = 5
        self.x0 = np.zeros(self.dim)
        self.max_iter = 1000
        self.sigma = 0.5
        self.seeds = [42, 123, 7, 2024, 999]
        self.results = {}  # (func_name, strategy_name) -> [fx, fx, ...]
        os.makedirs(RESULT_DIR, exist_ok=True)
        os.makedirs(CONVERGENCE_DIR, exist_ok=True)

    def run_optimizer(self, func, center_strategy, seed):
        params = CMAParameters.basic_from_literature(
            dim=self.dim, center_strategy=center_strategy, seed=seed
        )
        params.max_iter = self.max_iter
        params.sigma = self.sigma
        optimizer = cma_es(x0=self.x0, parameters=params)
        result = optimizer.optimize(func)
        # --- ZAPISZ KRZYWĄ ZBIEŻNOŚCI ---
        curve = getattr(optimizer, "f_history", None)
        if curve is None and hasattr(optimizer, "logger"):
            curve = getattr(optimizer.logger, "best_fitness_history", None)
        if curve is not None:
            curve_filename = f"{CONVERGENCE_DIR}/{func.__name__}_{type(center_strategy).__name__}_seed{seed}.csv"
            with open(curve_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "fx"])
                for i, fx_iter in enumerate(curve):
                    writer.writerow([i, fx_iter])
        return result

    def save_results_to_csv(self, filename=RESULTS_CSV):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["function", "strategy", "seed", "fx"])
            for (func_name, strategy_name), fx_list in self.results.items():
                for i, fx in enumerate(fx_list):
                    writer.writerow([func_name, strategy_name, self.seeds[i], fx])

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
        max_time = 5.0  # sekundy

        errors = []  # lista błędów

        for func_name, func in benchmark_functions.items():
            for strategy in strategies:
                fx_list = []
                for seed in self.seeds:
                    np.random.seed(seed)
                    start = time.time()
                    result = self.run_optimizer(func, strategy, seed)
                    elapsed = time.time() - start
                    fx = func(result)
                    fx_list.append(fx)
                    print(f"{func_name} {type(strategy).__name__}: f(x)={fx:.3e}, czas={elapsed:.2f}s")
                    if not (fx < tol and elapsed < max_time):
                        errors.append(f"{func_name} {type(strategy).__name__} seed={seed}: f(x)={fx}, czas={elapsed}s")
                self.results[(func_name, type(strategy).__name__)] = fx_list

        # Zapisz wyniki po teście
        self.save_results_to_csv(f"{RESULT_DIR}/{RESULTS_CSV}")
        self.save_results_pickle(f"{RESULT_DIR}/{RESULTS_PKL}")

        # Po wszystkim wypisz błędy i nie przerywaj testu
        if errors:
            print("\nNiespełnione przypadki:")
            for err in errors:
                print(err)

    def wilcoxon_center_on_function(self, func_name, strat1, strat2):
        fx1 = self.results.get((func_name, strat1))
        fx2 = self.results.get((func_name, strat2))
        if fx1 is None or fx2 is None:
            self.skipTest("Najpierw uruchom test_performance_and_collect, by zebrać wyniki.")
        # Sprawdź czy wszystkie różnice to zero
        diffs = np.array(fx1) - np.array(fx2)
        if np.allclose(diffs, 0):
            print(f"Wilcoxon {strat1} vs {strat2} on {func_name}: wszystkie wyniki identyczne, test pominięty.")
            return
        stat, p = wilcoxon(fx1, fx2)
        print(f"Wilcoxon {strat1} vs {strat2} on {func_name}: stat={stat}, p={p}")

    def test_wilcoxon_all(self):
        # Jeśli nie masz wyników w pamięci, wczytaj z pliku
        if not self.results:
            if os.path.exists(f"{RESULT_DIR}/{RESULTS_PKL}"):
                self.load_results_pickle(f"{RESULT_DIR}/{RESULTS_PKL}")
            else:
                self.skipTest("Brak wyników, najpierw uruchom test_performance_and_collect.")

        strategies = [
            "ArithmeticMeanCenterStrategy",
            "MedianCenterStrategy",
            "WeightedFitnessCenterStrategy",
            "WeightedRankCenterStrategy",
            "TrimmedMeanCenterStrategy",
        ]

        for func_name in benchmark_functions.keys():
            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    self.wilcoxon_center_on_function(
                        func_name,
                        strategies[i],
                        strategies[j]
                    )

if __name__ == "__main__":
    unittest.main()
