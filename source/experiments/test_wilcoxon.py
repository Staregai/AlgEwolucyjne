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
from scipy.stats import wilcoxon
import pickle
import os
import pandas as pd

RESULTS_CSV = "results/performance_results.csv"
RESULTS_PKL = "results/performance_results.pkl"
RESULT_DIR = "results"
CONVERGENCE_DIR = "results/convergence"

SEED_FILE = "experiments/seeds.txt"
class TestWilcoxon(unittest.TestCase):

    def setUp(self):
        self.dims = [2, 5, 10, 20]
        self.wilcoxon_logs = [] 

        with open(SEED_FILE) as f:
            seeds = [int(line.strip()) for line in f]

        self.seeds = seeds

    def load_results_pickle(self, filename=RESULTS_PKL):
        with open(filename, "rb") as f:
            self.results = pickle.load(f)

    def test_wilcoxon_all(self):
        # self.load_results_pickle()
        df = pd.read_csv(RESULTS_CSV)
        self.results = {}
        grouped = df.groupby(["function", "strategy", "dim"])
        for (func_name, strategy_name, dim), group in grouped:
            fx_list = group.sort_values("seed")["fx"].tolist()
            self.results[(func_name, strategy_name, int(dim))] = fx_list

        strategies = [
                "ArithmeticMeanCenterStrategy",
                "MedianCenterStrategy",
                "WeightedFitnessCenterStrategy",
                "WeightedRankCenterStrategy",
                "TrimmedMeanCenterStrategy",
            ]

        for func_name in benchmark_functions.keys():
            for dim in self.dims:
                for i in range(len(strategies)):
                    for j in range(i + 1, len(strategies)):
                        if (func_name in ["michalewicz", "booth"] and dim > 2):
                             continue
                        self.wilcoxon_center_on_function(
                                func_name, strategies[i], strategies[j], dim
                            )

        out_path = os.path.join(RESULT_DIR, "wilcoxon_results.csv")
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(
                [
                    "function",
                    "dim",
                    "strategy1",
                    "strategy2",
                    "statistic",
                    "p_value",
                    "significant",
                ]
            )
            writer.writerows(self.wilcoxon_logs)

    def wilcoxon_center_on_function(self, func_name, strat1, strat2, dim):
        fx1 = self.results.get((func_name, strat1, dim))
        fx2 = self.results.get((func_name, strat2, dim))
        if fx1 is None or fx2 is None:
            print(f"Brak wyników dla {func_name}, {strat1} lub {strat2}, dim={dim}")
            self.wilcoxon_logs.append([func_name, dim, strat1, strat2, "MISSING", "MISSING", "SKIPPED"])

        diffs = np.array(fx1) - np.array(fx2)
        if np.allclose(diffs, 0):
            print(
                f"Wilcoxon {strat1} vs {strat2} on {func_name} dim ={dim}: wszystkie wyniki identyczne, test pominięty."
            )
            self.wilcoxon_logs.append(
                [func_name, dim, strat1, strat2, "ALL_CLOSE", "ALL_CLOSE", "SKIPPED"]
            )
            return
        stat, p = wilcoxon(fx1, fx2)
        is_significant = "YES" if p < 0.05 else "NO"
        self.wilcoxon_logs.append([func_name, dim, strat1, strat2, stat, p, is_significant])  
        print(
            f"Wilcoxon {strat1} vs {strat2} on {func_name} dim={dim}: stat={stat}, p={p}"
        )


if __name__ == "__main__":
    unittest.main()
