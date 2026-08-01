import pandas as pd
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 60)

df = pd.read_csv(r"results\rules_search2_results.csv")
print("Total operaciones:", len(df))

def profit_factor(s):
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    return gp / gl if gl > 0 else float("inf")

g = df.groupby(["sl_pct", "tp_pct"]).agg(
    n_trades=("pnl_pct", "size"),
    win_rate=("pnl_pct", lambda s: round(100*(s>0).mean(), 1)),
    gross_profit=("pnl_pct", lambda s: round(s[s>0].sum(), 1)),
    gross_loss=("pnl_pct", lambda s: round(-s[s<0].sum(), 1)),
    total_pnl=("pnl_pct", lambda s: round(s.sum(), 1)),
).reset_index()
g["profit_factor"] = (g.gross_profit / g.gross_loss).round(3)
g_sorted = g.sort_values("profit_factor", ascending=False)

print("\n=== Ranking completo (30 combos) por Profit Factor ===")
print(g_sorted.to_string(index=False))

print("\n=== Referencia: sl_pct=0.05, tp_pct=0.10 (mejor avg_pnl de la corrida anterior) ===")
ref = g[(g.sl_pct==0.05)&(g.tp_pct==0.10)]
print(ref.to_string(index=False))

print("\n=== Efecto marginal de sl_pct en profit factor (promediando tp_pct) ===")
m1 = df.groupby("sl_pct")["pnl_pct"].apply(profit_factor).round(3)
print(m1)

print("\n=== Efecto marginal de tp_pct en profit factor (promediando sl_pct) ===")
m2 = df.groupby("tp_pct")["pnl_pct"].apply(profit_factor).round(3)
print(m2)

print("\n=== Top 10 por Profit Factor con n_trades >= 300 ===")
robust = g_sorted[g_sorted.n_trades >= 300]
print(robust.head(10).to_string(index=False))

print("\n=== Top 10 por Profit Factor con MENOS operaciones (n_trades < 350, favorece pocas ops) ===")
fewer = g_sorted[g_sorted.n_trades < 350].sort_values("profit_factor", ascending=False)
print(fewer.head(10).to_string(index=False))
