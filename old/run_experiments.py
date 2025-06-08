import numpy as np
import cma
from source.optimizer.cma_es import cma_es
import os
import csv
from source.functions.benchmark import benchmark_functions, benchmark_bounds
import tqdm
from source.optimizer.center_strategies import enter_strategies
from datetime import datetime

# PS python -m experiments.run_experiments

DIMENSIONS = [
          2, 
        #   5, 10,
            #    30, 50, 
            #    100
               ]
REPEATS = 1
OUTPUT_DIR = "source/experiments/results"
SEED_FILE = "source/experiments/seeds.txt"


os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(SEED_FILE) as f:
    seeds = [int(line.strip()) for line in f]

for func_name, func in benchmark_functions.items():
    for dim in DIMENSIONS:
        for actual_strategy in enter_strategies:
            for run in range(REPEATS):
                seed = seeds[run]
                bounds = benchmark_bounds.get(func_name, None)
                x0 = np.random.randn(dim)
                optimizer = cma_es(x0)
                optimizer.center_strategy = actual_strategy
                optimizer.logger.log_step = 10

                result = optimizer.optimize(func)
                today = datetime.now().strftime("%Y-%m-%d_%H-%M")
                result_file = f"{OUTPUT_DIR}/results_{func_name}_{dim}_{type(optimizer.center_strategy).__name__}_{today}.csv"
                with open(result_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([seed, result, dim])

                    writer.writerow([optimizer.logger.log_step])
                    writer.writerow(optimizer.logger.m_history)