import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from functions.benchmark import benchmark_functions

SAVE_DIR = "results/plots"
DIMS = [2, 5, 10, 20]


def plot_convergence_curves(
    curves_dir="results/convergence",
    func_name="rastrigin",
    strategies=None,
    save_dir="results/plots",
):
    if strategies is None:
        strategies = [
            "ArithmeticMeanCenterStrategy",
            "MedianCenterStrategy",
            "WeightedFitnessCenterStrategy",
            "WeightedRankCenterStrategy",
            "TrimmedMeanCenterStrategy",
        ]
    os.makedirs(save_dir, exist_ok=True)

    # osobne dla kazdego wymiaru
    for dim in DIMS:
        plt.figure(figsize=(10, 6))
        any_curve = False
        for strategy in strategies:
            all_curves = []
            if (func_name in ["michalewicz", "booth"] and dim > 2):
                continue
            
            files = glob.glob(f"{curves_dir}/{func_name}_{strategy}_dim{dim}_seed*.csv")
            if not files:
                print(
                    f"Brak plików dla strategii {strategy}, funkcji {func_name}, dim={dim}"
                )
            for file in files:
                df = pd.read_csv(file)
                if "fx" in df.columns and len(df["fx"]) > 0:
                    all_curves.append(df["fx"].values)
            if all_curves:
                any_curve = True
                min_len = min(len(curve) for curve in all_curves)
                all_curves = [curve[:min_len] for curve in all_curves]
                mean_curve = sum(all_curves) / len(all_curves)
                sorted_curves = sorted(all_curves, key=lambda curve: curve[-1])
                trimmed_curves = sorted_curves[: int(len(sorted_curves) * 0.9)]
                mean_curve = sum(trimmed_curves) / len(trimmed_curves)

                iteration_values = df["iteration"].values[:min_len]
                plt.plot(iteration_values, mean_curve, label=strategy)

        all_fx = []
        for line in plt.gca().get_lines():
            all_fx.extend(line.get_ydata())
        min_fx = min(all_fx)
        if min_fx > 0:
            plt.yscale("log")
            plt.ylim(max(min_fx, 1e-25), max(all_fx)) 
        else:
            plt.yscale("linear")
            print("Uwaga: Wartości <= 0, wykres na skali liniowej.")
        plt.xlabel("Iteracja")
        plt.ylabel("f(x)")
        plt.title(f"Krzywa zbieżności: {func_name}, dim={dim}")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"convergence_{func_name}_dim{dim}.png")
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        plt.close()



    # jeden wykres 4 subploty
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    for idx, dim in enumerate(DIMS):
        ax = axs[idx]
        any_curve = False
        for strategy in strategies:
            all_curves = []
            files = glob.glob(f"{curves_dir}/{func_name}_{strategy}_dim{dim}_seed*.csv")
            for file in files:
                df = pd.read_csv(file)
                if "fx" in df.columns and len(df["fx"]) > 0:
                    all_curves.append(df["fx"].values)
            if all_curves:
                any_curve = True
                min_len = min(len(curve) for curve in all_curves)
                all_curves = [curve[:min_len] for curve in all_curves]
                mean_curve = sum(all_curves) / len(all_curves)
                ax.plot(range(min_len), mean_curve, label=strategy)
        if any_curve:
            all_fx = []
            for line in ax.get_lines():
                all_fx.extend(line.get_ydata())
            min_fx = min(all_fx)
            if min_fx > 0:
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
            ax.set_title(f"dim={dim}")
            ax.set_xlabel("Iteracja")
            ax.set_ylabel("f(x)")
            ax.legend(fontsize=8)
        else:
            ax.set_title(f"dim={dim} (brak danych)")
            ax.axis("off")
    plt.suptitle(f"Krzywe zbieżności dla {func_name} (różne wymiary)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(save_dir, f"convergence_{func_name}_subplots.png")
    plt.savefig(plot_path)
    print(f"Saved subplot to {plot_path}")
    plt.close()


if __name__ == "__main__":
    functions = [
        "sphere",
        "rosenbrock",
        "rastrigin",
        "ellipsoid", 
        "ackley",
        "schwefel",
        "griewank",
        "zakharov",
        "michalewicz",
        "booth"
    ]

    for func in functions:
        plot_convergence_curves(func_name=func)
