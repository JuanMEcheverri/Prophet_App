import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

df = pd.read_csv(r"results\results.csv")
print("filas totales:", len(df))
print("tickers:", df['ticker'].nunique(), " cutoffs:", df['cutoff'].nunique(), " windows:", df['window'].unique())

WINDOW_ORDER = ["6 meses", "1 año", "2 años", "3 años", "4 años"]

print("\n=== 1) MAPE 0-30d: media vs mediana por ventana (para ver si 6 meses esta sesgado por outliers) ===")
g = df.groupby("window")["mape_0_30d"].agg(["mean", "median", "std", "max"]).reindex(WINDOW_ORDER)
print(g.round(2))

print("\n=== 2) Cuantos casos de 6 meses tienen MAPE > 30% (outliers tipo AAPL) ===")
for w in WINDOW_ORDER:
    sub = df[df.window == w]
    n_outlier = (sub["mape_0_30d"] > 30).sum()
    print(f"{w}: {n_outlier}/{len(sub)} casos con MAPE>30%  ({100*n_outlier/len(sub):.0f}%)")

print("\n=== 3) Error absoluto por checkpoint (mediana, robusta a outliers) ===")
cps = [5, 10, 15, 20, 25, 30]
med_table = pd.DataFrame(index=WINDOW_ORDER)
for cp in cps:
    col = f"abs_err_{cp}d"
    med_table[f"+{cp}d"] = df.groupby("window")[col].median().reindex(WINDOW_ORDER)
print(med_table.round(2))

print("\n=== 3b) Error absoluto por checkpoint (MEDIA, para contrastar con la mediana) ===")
mean_table = pd.DataFrame(index=WINDOW_ORDER)
for cp in cps:
    col = f"abs_err_{cp}d"
    mean_table[f"+{cp}d"] = df.groupby("window")[col].mean().reindex(WINDOW_ORDER)
print(mean_table.round(2))

print("\n=== 4) Hit-rate de la tesis: subio 5% real vs prophet lo predijo (max yhat 30d >= +5%) ===")
hit = df.groupby("window").agg(
    pct_actual_hit=("actual_hit_5pct_30d", "mean"),
    pct_pred_hit=("yhat_predicted_hit_5pct_30d", "mean"),
).reindex(WINDOW_ORDER)
hit["pct_actual_hit"] *= 100
hit["pct_pred_hit"] *= 100
print(hit.round(1))

print("\n=== 5) Precision/Recall de la señal 'Prophet predice +5% en 30d' vs realidad ===")
for w in WINDOW_ORDER:
    sub = df[df.window == w]
    tp = ((sub.yhat_predicted_hit_5pct_30d == True) & (sub.actual_hit_5pct_30d == True)).sum()
    fp = ((sub.yhat_predicted_hit_5pct_30d == True) & (sub.actual_hit_5pct_30d == False)).sum()
    fn = ((sub.yhat_predicted_hit_5pct_30d == False) & (sub.actual_hit_5pct_30d == True)).sum()
    tn = ((sub.yhat_predicted_hit_5pct_30d == False) & (sub.actual_hit_5pct_30d == False)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    n_signals = tp + fp
    print(f"{w:8s}: señales={n_signals:4d}  precision={precision*100:5.1f}%  recall={recall*100:5.1f}%  (tp={tp} fp={fp} fn={fn} tn={tn})")

print("\n=== 6) Bias medio (signo) por ventana -- sobreestima o subestima? ===")
bias = df.groupby("window")["bias_0_30d"].agg(["mean","median"]).reindex(WINDOW_ORDER)
print(bias.round(2))

print("\n=== 7) Ranking de ventanas por ticker (cual ventana gana mas seguido, por MAPE mediano) ===")
pivot = df.pivot_table(index=["ticker","cutoff"], columns="window", values="mape_0_30d")
pivot = pivot[WINDOW_ORDER]
wins = pivot.idxmin(axis=1).value_counts().reindex(WINDOW_ORDER).fillna(0).astype(int)
print(wins)
print(f"total comparaciones: {len(pivot)}")
