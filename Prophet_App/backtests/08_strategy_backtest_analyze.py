import pandas as pd
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 120)

df = pd.read_csv(r"results\trades.csv")
WINDOW_ORDER = ["6 meses", "1 año", "2 años", "3 años", "4 años"]

print("Total operaciones simuladas:", len(df))
print("Tickers distintos con al menos 1 trade:", df.ticker.nunique())

print("\n=== 1) Metricas por ventana de entrenamiento ===")
rows = []
for w in WINDOW_ORDER:
    sub = df[df.window == w]
    n = len(sub)
    win_rate = 100 * (sub.pnl_pct > 0).mean()
    avg_pnl = sub.pnl_pct.mean()
    med_pnl = sub.pnl_pct.median()
    vc = sub.exit_reason.value_counts(normalize=True).mul(100)
    rows.append({
        "ventana": w, "n_trades": n, "win_rate%": round(win_rate,1),
        "avg_pnl%": round(avg_pnl,2), "median_pnl%": round(med_pnl,2),
        "%TP": round(vc.get("TP",0),1), "%SL": round(vc.get("SL",0),1), "%MAX_HOLD": round(vc.get("MAX_HOLD",0),1),
    })
res = pd.DataFrame(rows).set_index("ventana")
print(res)

print("\n=== 2) P&L acumulado simple (suma de pnl_pct, no compuesto) por ventana ===")
print(df.groupby("window")["pnl_pct"].sum().reindex(WINDOW_ORDER).round(1))

print("\n=== 3) Expectativa por operacion (avg_pnl%) -- ordenado de mejor a peor ===")
print(res.sort_values("avg_pnl%", ascending=False)[["n_trades","win_rate%","avg_pnl%"]])

# ---- Analisis por ticker, usando la ventana recomendada "1 año" ----
print("\n\n=== 4) Por ticker (ventana = 1 año) ===")
sub1 = df[df.window == "1 año"]
per_ticker = sub1.groupby("ticker").agg(
    n_trades=("pnl_pct","size"),
    win_rate=("pnl_pct", lambda s: 100*(s>0).mean()),
    avg_pnl=("pnl_pct","mean"),
    median_pnl=("pnl_pct","median"),
    sl_rate=("exit_reason", lambda s: 100*(s=="SL").mean()),
    tp_rate=("exit_reason", lambda s: 100*(s=="TP").mean()),
)
per_ticker = per_ticker.round(1)
per_ticker_min3 = per_ticker[per_ticker.n_trades >= 3].sort_values("avg_pnl")

print(f"Tickers con >=3 trades (ventana 1 año): {len(per_ticker_min3)}")
print("\n--- 15 PEORES (menor avg_pnl%) -- candidatos a corregir/excluir ---")
print(per_ticker_min3.head(15))
print("\n--- 15 MEJORES (mayor avg_pnl%) ---")
print(per_ticker_min3.sort_values("avg_pnl", ascending=False).head(15))

print("\n--- Tickers con SL rate >= 50% (bandera de riesgo alto) ---")
risky_sl = per_ticker_min3[per_ticker_min3.sl_rate >= 50].sort_values("sl_rate", ascending=False)
print(risky_sl)

print("\n--- Tickers con avg_pnl NEGATIVO (pierden dinero en promedio) ---")
losers = per_ticker_min3[per_ticker_min3.avg_pnl < 0].sort_values("avg_pnl")
print(f"n={len(losers)}")
print(losers)

per_ticker.to_csv(r"results\per_ticker_strategy.csv")
res.to_csv(r"results\per_window_strategy.csv")
print("\nGuardado per_ticker_strategy.csv y per_window_strategy.csv")
