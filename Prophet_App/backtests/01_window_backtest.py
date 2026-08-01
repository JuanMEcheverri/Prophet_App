"""
Backtest walk-forward: que ventana de entrenamiento (t_start) hace que el
yhat de Prophet sea mas preciso DENTRO de los primeros 30 dias futuros,
usando exactamente los mismos hiperparametros que Prophet_App/app.py.

La estrategia real solo le importa que pase en <=30 dias (thesis: el precio
sube ~5% antes de 30 dias), asi que NO evaluamos mas alla de ese horizonte.

Metodologia:
- Para cada ticker, se descarga UNA vez el historico completo.
- Para cada (ventana, fecha de corte): se entrena Prophet SOLO con datos
  hasta la fecha de corte (nunca ve el futuro), se proyecta ~33 dias, y se
  compara yhat contra el precio real observado despues del corte
  (verdadero out-of-sample) en VARIOS puntos de corte dentro de la ventana
  de 30 dias: +5, +10, +15, +20, +25, +30 dias.
"""
import time, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet

TODAY = pd.Timestamp("2026-08-01")

# Nasdaq-100 completo (mismo universo que NASDAQ100 en Prophet_App/app.py),
# no solo la seleccion por defecto -- analisis expandido a todas las acciones.
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","TMUS","ADBE","CSCO","PEP","QCOM","INTU","AMAT",
    "TXN","ISRG","AMGN","BKNG","MU","VRTX","REGN","PANW","KLAC","GILD",
    "LRCX","ADI","SBUX","MDLZ","MELI","SNPS","CDNS","CTAS","PAYX","ROST",
    "DDOG","ABNB","PYPL","MAR","ORLY","FAST","IDXX","WDAY","PCAR","AEP",
    "FTNT","CPRT","MNST","BIIB","DLTR","TTD","DXCM","EBAY","ENPH","WBD",
    "XEL","EXC","EA","ODFL","CRWD","NXPI","ON","ZS","VRSK","GEHC","HON",
    "FSLR","CEG","GFS","PDD","NTAP","APP","SMCI","ZBRA","KHC","ILMN",
    "OKTA","BMRN","MDB","ZM","SWKS","TEAM","MAR","MRVL","ROP","ANSS",
]
TICKERS = sorted(set(TICKERS))

# Umbrales para clasificar el resultado REAL de cada trade hipotetico a +30d
# (independiente de si Prophet predijo o no la subida):
#   SUBIO_5+   -> precio real subio 5% o mas respecto al precio del corte
#   BAJO_-2    -> precio real bajo mas de 2% respecto al precio del corte
#   MEDIO      -> quedo entre esos dos umbrales
OUTCOME_UP_PCT = 5.0
OUTCOME_DOWN_PCT = -2.0

WINDOWS = {
    "6 meses": pd.DateOffset(months=6),
    "1 año":   pd.DateOffset(years=1),
    "2 años":  pd.DateOffset(years=2),
    "3 años":  pd.DateOffset(years=3),
    "4 años":  pd.DateOffset(years=4),
}

# Solo nos interesa el horizonte de 30 dias (tesis: sube ~5% antes de 30 dias).
# Se pide un poco mas (33) para tener margen y asegurar que exista un dia
# de mercado real cerca de +30.
FORECAST_PERIODS = 33
ACTUAL_WINDOW_DAYS = 33
CHECKPOINTS = [5, 10, 15, 20, 25, 30]

# 10 cortes walk-forward espaciados ~55 dias, terminando >=33 dias antes de hoy
N_CUTOFFS = 10
STEP_DAYS = 55
last_cutoff = TODAY - pd.Timedelta(days=ACTUAL_WINDOW_DAYS)
CUTOFFS = [last_cutoff - pd.Timedelta(days=STEP_DAYS * i) for i in range(N_CUTOFFS)]

DOWNLOAD_START = min(c - w for c in CUTOFFS for w in WINDOWS.values()).strftime("%Y-%m-%d")
DOWNLOAD_END = TODAY.strftime("%Y-%m-%d")

print(f"Tickers: {len(TICKERS)}", flush=True)
print(f"Cutoffs: {[c.strftime('%Y-%m-%d') for c in CUTOFFS]}", flush=True)
print(f"Checkpoints (dias): {CHECKPOINTS}", flush=True)
print(f"Descarga: {DOWNLOAD_START} -> {DOWNLOAD_END}", flush=True)

t0 = time.time()
raw = yf.download(TICKERS, start=DOWNLOAD_START, end=DOWNLOAD_END, auto_adjust=True, progress=False)
print(f"Descarga completa en {time.time()-t0:.1f}s", flush=True)

close_all = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": TICKERS[0]})

series = {}
for t in TICKERS:
    try:
        cl = close_all[t].dropna()
        if cl.index.tz is not None:
            cl.index = cl.index.tz_localize(None)
        cl.index = cl.index.normalize()
        series[t] = cl
    except Exception as e:
        print(f"skip {t}: {e}", flush=True)

results = []
total = len(series) * len(WINDOWS) * len(CUTOFFS)
done = 0
t_start_all = time.time()

for ticker, cl in series.items():
    for cutoff in CUTOFFS:
        cutoff_px_series = cl[cl.index <= cutoff]
        if cutoff_px_series.empty:
            continue
        cutoff_px = cutoff_px_series.iloc[-1]

        for wname, wdelta in WINDOWS.items():
            done += 1
            train_start = cutoff - wdelta
            train_mask = (cl.index > train_start) & (cl.index <= cutoff)
            train = cl[train_mask]

            if len(train) < 60:
                continue

            df_fit = pd.DataFrame({"ds": train.index, "y": train.values})

            try:
                m = Prophet(
                    interval_width=0.95, seasonality_mode="multiplicative", growth="linear",
                    changepoint_prior_scale=0.05, daily_seasonality=False,
                    yearly_seasonality=True, weekly_seasonality=False,
                )
                m.fit(df_fit)
                future = m.make_future_dataframe(periods=FORECAST_PERIODS)
                pred = m.predict(future)[["ds", "yhat"]].set_index("ds")
            except Exception as e:
                print(f"fit error {ticker} {cutoff.date()} {wname}: {str(e)[:80]}", flush=True)
                continue

            future_actual = cl[(cl.index > cutoff) & (cl.index <= cutoff + pd.Timedelta(days=ACTUAL_WINDOW_DAYS))]
            if future_actual.empty:
                continue

            errs = []
            for d, actual_px in future_actual.items():
                if d not in pred.index:
                    continue
                yhat_v = pred.loc[d, "yhat"]
                if yhat_v <= 0:
                    continue
                pct_err = (actual_px / yhat_v - 1) * 100
                errs.append((d, pct_err))

            if not errs:
                continue

            errs_df = pd.DataFrame(errs, columns=["ds", "pct_err"])
            mape_0_30 = errs_df["pct_err"].abs().mean()
            bias_0_30 = errs_df["pct_err"].mean()

            row = {
                "ticker": ticker, "cutoff": cutoff.strftime("%Y-%m-%d"), "window": wname,
                "train_rows": len(train), "n_obs": len(errs_df),
                "mape_0_30d": mape_0_30, "bias_0_30d": bias_0_30,
            }

            for cp in CHECKPOINTS:
                target_date = cutoff + pd.Timedelta(days=cp)
                nearest_idx = (errs_df["ds"] - target_date).abs().idxmin()
                nearest_date = errs_df.loc[nearest_idx, "ds"]
                # ignorar si el dia real mas cercano queda a mas de 4 dias del checkpoint
                if abs((nearest_date - target_date).days) > 4:
                    row[f"err_{cp}d"] = None
                    row[f"abs_err_{cp}d"] = None
                else:
                    v = float(errs_df.loc[nearest_idx, "pct_err"])
                    row[f"err_{cp}d"] = v
                    row[f"abs_err_{cp}d"] = abs(v)

            # Metrica alineada a la tesis: alcanzo el precio real +5% en algun
            # momento dentro de los 30 dias? y lo predijo yhat (su maximo)?
            actual_hit_5pct = bool((future_actual / cutoff_px - 1 >= 0.05).any())
            pred_future_only = pred[(pred.index > cutoff) & (pred.index <= cutoff + pd.Timedelta(days=30))]
            yhat_max_dev = float((pred_future_only["yhat"] / cutoff_px - 1).max() * 100) if not pred_future_only.empty else None
            row["actual_hit_5pct_30d"] = actual_hit_5pct
            row["yhat_predicted_hit_5pct_30d"] = (yhat_max_dev is not None and yhat_max_dev >= 5)
            row["yhat_max_dev_pct_30d"] = yhat_max_dev

            # Resultado REAL a +30d respecto al precio del corte (no respecto a
            # yhat) -- clasificacion en 3 categorias para saber que tan seguido
            # una señal de "sube 5%" termina en realidad en una caida fuerte.
            target_date_30 = cutoff + pd.Timedelta(days=30)
            nearest_30_idx = (future_actual.index - target_date_30).map(lambda d: abs(d.days)).argmin()
            nearest_30_date = future_actual.index[nearest_30_idx]
            if abs((nearest_30_date - target_date_30).days) <= 4:
                actual_px_30 = future_actual.iloc[nearest_30_idx]
                actual_chg_30d_pct = float((actual_px_30 / cutoff_px - 1) * 100)
                if actual_chg_30d_pct >= OUTCOME_UP_PCT:
                    outcome_bucket = "SUBIO_5+"
                elif actual_chg_30d_pct <= OUTCOME_DOWN_PCT:
                    outcome_bucket = "BAJO_-2"
                else:
                    outcome_bucket = "MEDIO"
            else:
                actual_chg_30d_pct = None
                outcome_bucket = None
            row["actual_chg_30d_pct"] = actual_chg_30d_pct
            row["outcome_bucket"] = outcome_bucket

            results.append(row)

    elapsed = time.time() - t_start_all
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    print(f"[{done}/{total}] ticker={ticker} completado -- {elapsed:.0f}s transcurridos, ~{remaining:.0f}s restantes", flush=True)

df_results = pd.DataFrame(results)
out_path = r"results\results.csv"
df_results.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path}  ({len(df_results)} filas)", flush=True)
print(f"Tiempo total: {time.time()-t_start_all:.0f}s", flush=True)
