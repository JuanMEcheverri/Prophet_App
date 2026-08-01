import pandas as pd
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 160)

df = pd.read_csv(r"results\rules_search_results.csv")
print("Total operaciones:", len(df))

PARAMS = ["entry_dev", "sl_pct", "tp_pct", "max_hold"]

print("\n=== Ranking completo por combo (144 filas) ===")
g = df.groupby(PARAMS).agg(
    n_trades=("pnl_pct", "size"),
    win_rate=("pnl_pct", lambda s: round(100*(s>0).mean(), 1)),
    avg_pnl=("pnl_pct", lambda s: round(s.mean(), 2)),
    median_pnl=("pnl_pct", lambda s: round(s.median(), 2)),
).reset_index()
g_sorted = g.sort_values("avg_pnl", ascending=False)
print(g_sorted.head(20).to_string(index=False))

print("\n--- Baseline actual (entry_dev=0.03, sl_pct=0.03, tp_pct=0.05, max_hold=30) ---")
base = g[(g.entry_dev==0.03)&(g.sl_pct==0.03)&(g.tp_pct==0.05)&(g.max_hold==30)]
print(base.to_string(index=False))
print("Ranking del baseline por avg_pnl (de 144):", (g_sorted.avg_pnl > base.avg_pnl.values[0]).sum()+1)

print("\n=== Efecto marginal de cada parametro (promediando los otros 3) ===")
for p in PARAMS:
    print(f"--- {p} ---")
    mg = df.groupby(p).agg(
        n=("pnl_pct","size"), win_rate=("pnl_pct", lambda s: round(100*(s>0).mean(),1)),
        avg_pnl=("pnl_pct", lambda s: round(s.mean(),2)), median_pnl=("pnl_pct", lambda s: round(s.median(),2)),
    )
    print(mg)
    print()

print("=== Top 10 por avg_pnl con n_trades >= 300 (evitar ganadores con poca muestra) ===")
robust = g_sorted[g_sorted.n_trades >= 300]
print(robust.head(10).to_string(index=False))

print("\n=== Win-rate cabeza a cabeza entre los 144 combos ===")
pivot = df.pivot_table(index=["ticker","cutoff"], columns=PARAMS, values="pnl_pct", aggfunc="first")
# Nota: pivot puede tener NaN donde una combinacion no genero trade para ese (ticker,cutoff)
wins = pivot.idxmax(axis=1, skipna=True).value_counts()
print("Top 10 combos que mas veces ganan cabeza a cabeza:")
print(wins.head(10))
