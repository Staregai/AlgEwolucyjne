import numpy as np
import cma
import os
import csv
from benchmark_functions import benchmark_functions
import tqdm

DIMENSIONS = [2, 5, 10, 30, 50, 200]
REPEATS = 10
OUTPUT_DIR = "results"
SEED_FILE = "seeds.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(SEED_FILE) as f:
    seeds = [int(line.strip()) for line in f]

for func_name, func in benchmark_functions.items():
    for dim in DIMENSIONS:
        for run in range(REPEATS):
            seed = seeds[run]
            np.random.seed(seed)
            x0 = np.random.randn(dim)
            sigma = 0.5
            es = cma.CMAEvolutionStrategy(x0, sigma, {'seed': seed})
            result = es.optimize(func)
            best_f = result.result.fbest

            result_file = f"{OUTPUT_DIR}/results_{func_name}_{dim}.csv"
            with open(result_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([seed, best_f, dim])
