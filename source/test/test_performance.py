import unittest
import numpy as np
from optimizer.cma_es import cma_es
from functions.benchmark import benchmark_functions
import optimizer.center_strategies as mn
from optimizer.cma_parameters import CMAParameters
import time
from scipy.stats import wilcoxon
import csv
import os
import pickle
from tqdm import tqdm

RESULTS_CSV = "performance_results.csv"
RESULTS_PKL = "performance_results.pkl"
CONV_CURVES_DIR = "convergence_curves"
SUMMARY_TXT = "performance_summary.txt"

class TestCMAESPerformance(unittest.TestCase):
    def setUp(self):
        self.dims = [2, 5, 10, 20]
        self.max_iter = 2000
        self.sigma = 0.5
        self.seeds = [42, 123, 7, 2024, 999]
        self.results = {}  # (func_name, strategy_name, dim) -> [fx, fx, ...]
        os.makedirs(CONV_CURVES_DIR, exist_ok=True)

    def run_optimizer(self, func, center_strategy, seed, dim):
        x0 = np.zeros(dim)
        params = CMAParameters.basic_from_literature(
            dim=dim, center_strategy=center_strategy, seed=seed
        )
        params.max_iter = self.max_iter
        params.sigma = self.sigma
        optimizer = cma_es(x0=x0, parameters=params)
        result = optimizer.optimize(func)
        # --- ZAPISZ KRZYWĄ ZBIEŻNOŚCI ---
        curve = getattr(optimizer, "f_history", None)
        if curve is None and hasattr(optimizer, "logger"):
            curve = getattr(optimizer.logger, "best_fitness_history", None)
        if curve is not None:
            curve_filename = f"{CONV_CURVES_DIR}/{func.__name__}_{type(center_strategy).__name__}_dim{dim}_seed{seed}.csv"
            with open(curve_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "fx"])
                for i, fx_iter in enumerate(curve):
                    writer.writerow([i, fx_iter])
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
        max_time = 10.0  # sekundy

        errors = []

        with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
            for func_name, func in benchmark_functions.items():
                for dim in self.dims:
                    for strategy in tqdm(strategies, desc=f"Testing {func_name} dim={dim}"):
                        fx_list = []
                        for seed in self.seeds:
                            np.random.seed(seed)
                            start = time.time()
                            result = self.run_optimizer(func, strategy, seed, dim)
                            elapsed = time.time() - start
                            fx = func(result)
                            fx_list.append(fx)
                            line = f"{func_name} dim={dim} {type(strategy).__name__}: f(x)={fx:.3e}, czas={elapsed:.2f}s"
                            f.write(line + "\n")
                            if not (fx < tol and elapsed < max_time):
                                errors.append(f"{func_name} {type(strategy).__name__} dim={dim} seed={seed}: f(x)={fx}, czas={elapsed}s")
                        self.results[(func_name, type(strategy).__name__, dim)] = fx_list

            self.save_results_to_csv()
            self.save_results_pickle()

            if errors:
                f.write("\nNiespełnione przypadki:\n")
                for err in errors:
                    f.write(err + "\n")

            # --- PODSUMOWANIE: ŚREDNIA I ODCHYLENIE STANDARDOWE ---
            f.write("\nPodsumowanie: Śr. końcowa wartość i Odch. std. dla każdej kombinacji (funkcja, strategia, wymiar):\n")
            f.write("Funkcja | Strategia | Wymiar | Śr. końcowa wartość | Odch. std.\n")
            for (func_name, strategy_name, dim), fx_list in self.results.items():
                mean_fx = np.mean(fx_list)
                std_fx = np.std(fx_list)
                f.write(f"{func_name} | {strategy_name} | {dim} | {mean_fx:.2e} | {std_fx:.2e}\n")

    def wilcoxon_center_on_function(self, func_name, strat1, strat2, dim):
        fx1 = self.results.get((func_name, strat1, dim))
        fx2 = self.results.get((func_name, strat2, dim))
        if fx1 is None or fx2 is None:
            return f"Brak wyników dla {func_name}, {strat1} lub {strat2}, dim={dim}\n"
        diffs = np.array(fx1) - np.array(fx2)
        if np.allclose(diffs, 0):
            return f"Wilcoxon {strat1} vs {strat2} on {func_name} dim={dim}: wszystkie wyniki identyczne, test pominięty.\n"
        stat, p = wilcoxon(fx1, fx2)
        return f"Wilcoxon {strat1} vs {strat2} on {func_name} dim={dim}: stat={stat}, p={p}\n"

    def test_wilcoxon_all(self):
        if not self.results:
            if os.path.exists(RESULTS_PKL):
                self.load_results_pickle()
            else:
                self.skipTest("Brak wyników, najpierw uruchom test_performance_and_collect.")

        strategies = [
            "ArithmeticMeanCenterStrategy",
            "MedianCenterStrategy",
            "WeightedFitnessCenterStrategy",
            "WeightedRankCenterStrategy",
            "TrimmedMeanCenterStrategy",
        ]
        with open("wilcoxon_summary.txt", "w", encoding="utf-8") as f:
            for func_name in benchmark_functions.keys():
                for dim in self.dims:
                    for i in range(len(strategies)):
                        for j in range(i + 1, len(strategies)):
                            result = self.wilcoxon_center_on_function(
                                func_name,
                                strategies[i],
                                strategies[j],
                                dim
                            )
                            f.write(result)

if __name__ == "__main__":
    unittest.main()