import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from functions.benchmark import benchmark_functions

SAVE_DIR = "plots"

def plot_convergence_curves(curves_dir="convergence_curves", func_name="rastrigin", strategies=None):
    if strategies is None:
        strategies = [
            "ArithmeticMeanCenterStrategy",
            "MedianCenterStrategy",
            "WeightedFitnessCenterStrategy",
            "WeightedRankCenterStrategy",
            "TrimmedMeanCenterStrategy",
        ]
    plt.figure(figsize=(10, 6))
    for strategy in strategies:
        all_curves = []
        files = glob.glob(f"{curves_dir}/{func_name}_{strategy}_seed*.csv")
        for file in files:
            df = pd.read_csv(file)
            all_curves.append(df["fx"].values)
        if all_curves:
            min_len = min(len(curve) for curve in all_curves)
            all_curves = [curve[:min_len] for curve in all_curves]
            mean_curve = sum(all_curves) / len(all_curves)
            plt.plot(range(min_len), mean_curve, label=strategy)
    plt.yscale("log")
    plt.xlabel("Iteracja")
    plt.ylabel("f(x)")
    plt.title(f"Krzywe zbieżności dla {func_name}")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f"convergence_{func_name}.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.show()
  
if __name__ == "__main__":
    for func_name in benchmark_functions.keys():
        print(f"Plotting convergence curves for {func_name}")
        plot_convergence_curves(func_name=func_name)
    print("All convergence curves plotted.")