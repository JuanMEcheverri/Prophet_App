import pandas as pd
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 120)

df = pd.read_csv(r"results\results.csv")
print("filas totales:", len(df))
print("tickers:", df['ticker'].nunique(), " cutoffs:", df['cutoff'].nunique())

WINDOW_ORDER = ["6 meses", "1 año", "2 años", "3 años", "4 años"]

print("\n=== 1) MAPE 0-30d: media vs mediana por ventana (universo completo, 89 tickers) ===")
g = df.groupby("window")["mape_0_30d"].agg(["mean", "median", "std"]).reindex(WINDOW_ORDER)
print(g.round(2))

print("\n=== 2) Tasa de outliers (MAPE>30%) por ventana ===")
for w in WINDOW_ORDER:
    sub = df[df.window == w]
    n_outlier = (sub["mape_0_30d"] > 30).sum()
    print(f"{w}: {n_outlier}/{len(sub)} ({100*n_outlier/len(sub):.1f}%)")

print("\n=== 3) Error mediano por checkpoint ===")
cps = [5,10,15,20,25,30]
med_table = pd.DataFrame(index=WINDOW_ORDER)
for cp in cps:
    med_table[f"+{cp}d"] = df.groupby("window")[f"abs_err_{cp}d"].median().reindex(WINDOW_ORDER)
print(med_table.round(2))

print("\n=== 4) Win-rate (menor MAPE cabeza a cabeza) ===")
pivot = df.pivot_table(index=["ticker","cutoff"], columns="window", values="mape_0_30d")
pivot = pivot[WINDOW_ORDER]
wins = pivot.idxmin(axis=1).value_counts().reindex(WINDOW_ORDER).fillna(0).astype(int)
print(wins, f"\ntotal comparaciones: {len(pivot)}")

print("\n=== 5) De las señales 'Prophet predice +5% en 30d' -- que paso REALMENTE (por ventana) ===")
for w in WINDOW_ORDER:
    sub = df[(df.window==w) & (df.yhat_predicted_hit_5pct_30d==True) & (df.outcome_bucket.notna())]
    n = len(sub)
    if n == 0:
        print(f"{w}: sin señales")
        continue
    vc = sub["outcome_bucket"].value_counts()
    subio = vc.get("SUBIO_5+",0); bajo = vc.get("BAJO_-2",0); medio = vc.get("MEDIO",0)
    print(f"{w:8s}: n={n:4d}  SUBIO_5+={subio:4d} ({100*subio/n:5.1f}%)  BAJO_-2={bajo:4d} ({100*bajo/n:5.1f}%)  MEDIO={medio:4d} ({100*medio/n:5.1f}%)")

print("\n=== 6) Lo mismo pero SOLO con ventana de 1 año (la recomendada) ===")
sub1 = df[(df.window=="1 año") & (df.yhat_predicted_hit_5pct_30d==True) & (df.outcome_bucket.notna())]
print(f"Total señales con 1 año: {len(sub1)}")
print(sub1["outcome_bucket"].value_counts())
print(sub1["outcome_bucket"].value_counts(normalize=True).mul(100).round(1))

print("\n=== 7) Distribucion global de outcome_bucket (TODOS los casos, prediga o no Prophet) por ventana ===")
for w in WINDOW_ORDER:
    sub = df[(df.window==w) & (df.outcome_bucket.notna())]
    vc = sub["outcome_bucket"].value_counts(normalize=True).mul(100).round(1)
    print(f"{w}: {vc.to_dict()}")

print("\n=== 8) Por ticker (ventana=1 año), entre las señales de compra: cuantas veces SUBIO vs BAJO vs MEDIO ===")
sub1t = df[(df.window=="1 año") & (df.yhat_predicted_hit_5pct_30d==True) & (df.outcome_bucket.notna())]
per_ticker = sub1t.groupby(["ticker","outcome_bucket"]).size().unstack(fill_value=0)
for col in ["SUBIO_5+","BAJO_-2","MEDIO"]:
    if col not in per_ticker.columns:
        per_ticker[col] = 0
per_ticker["n_señales"] = per_ticker[["SUBIO_5+","BAJO_-2","MEDIO"]].sum(axis=1)
per_ticker = per_ticker[per_ticker["n_señales"] >= 3]  # al menos 3 señales para ser informativo
per_ticker["pct_SUBIO"] = (100*per_ticker["SUBIO_5+"]/per_ticker["n_señales"]).round(0)
per_ticker["pct_BAJO"] = (100*per_ticker["BAJO_-2"]/per_ticker["n_señales"]).round(0)
per_ticker = per_ticker.sort_values("pct_BAJO", ascending=False)
print(f"tickers con >=3 señales de compra (1 año): {len(per_ticker)}")
print("\n--- TOP 15 mas riesgosos (mayor % BAJO_-2) ---")
print(per_ticker[["n_señales","SUBIO_5+","BAJO_-2","MEDIO","pct_SUBIO","pct_BAJO"]].head(15))
print("\n--- TOP 15 mas confiables (mayor % SUBIO_5+, entre los que tienen >=3 señales) ---")
print(per_ticker.sort_values("pct_SUBIO", ascending=False)[["n_señales","SUBIO_5+","BAJO_-2","MEDIO","pct_SUBIO","pct_BAJO"]].head(15))

per_ticker.to_csv(r"results\per_ticker_outcomes.csv")
print("\nGuardado per_ticker_outcomes.csv")
