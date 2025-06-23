import pandas as pd

# Wczytaj plik CSV
df = pd.read_csv("results/performance_results.csv")

# Grupuj po funkcji, strategii i wymiarze
grouped = df.groupby(["function", "strategy", "dim"])

# Przygotuj dane do nowego DataFrame
rows = []
for (func, strat, dim), group in grouped:
    mean_fx = group["fx"].mean()
    std_fx = group["fx"].std()
    rows.append({
        "function": func,
        "strategy": strat,
        "dim": dim,
        "mean_fx": mean_fx,
        "std_fx": std_fx
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv("results/convergence_summary_stats.csv", index=False)
print("Zapisano podsumowanie statystyk do 'performance_summary_stats.csv'")