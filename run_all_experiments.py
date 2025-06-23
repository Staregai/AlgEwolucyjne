import subprocess
import sys
import os


def run_command(command, desc):
    print(desc)
    result = subprocess.run(command, shell=True, cwd="source")
    if result.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) < 2:
        print("Usage: python run_all.py [flags]")
        print("Flags: e = experiments, w = wilcoxon, v = visualization")
        sys.exit(1)

    flags = sys.argv[1]

    if "e" in flags:
        run_command(
            "python -m experiments.test_performance_convergence",
            "Running performance and convergence tests",
        )

    if "w" in flags:
        run_command(
            "python -m experiments.test_wilcoxon", "Running Wilcoxon statistical tests"
        )

    if "v" in flags:
        run_command("python -m experiments.visualizer", "Generating plots")
