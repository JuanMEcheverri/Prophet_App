"""
Fase de validacion: correr el combo ganador del grid search
(changepoint_prior_scale=0.05, seasonality_mode="additive", yearly_seasonality=True)
contra el universo COMPLETO (89 tickers x 10 cortes), ventana fija en "1 año",
para confirmar que la mejora se sostiene a escala completa (no solo en la
muestra reducida de 40 tickers x 6 cortes de la busqueda).
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

WINDOW = pd.DateOffset(years=1)
WINNING_COMBO = {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive", "yearly_seasonality": True}

FORECAST_PERIODS = 33
ACTUAL_WINDOW_DAYS = 33
CHECKPOINTS = [5, 10, 15, 20, 25, 30]

N_CUTOFFS = 10
STEP_DAYS = 55
last_cutoff = TODAY - pd.Timedelta(days=ACTUAL_WINDOW_DAYS)
CUTOFFS = [last_cutoff - pd.Timedelta(days=STEP_DAYS * i) for i in range(N_CUTOFFS)]

DOWNLOAD_START = min(c - WINDOW for c in CUTOFFS).strftime("%Y-%m-%d")
DOWNLOAD_END = TODAY.strftime("%Y-%m-%d")

print(f"Tickers: {len(TICKERS)}", flush=True)
print(f"Cutoffs: {[c.strftime('%Y-%m-%d') for c in CUTOFFS]}", flush=True)
print(f"Combo a validar: {WINNING_COMBO}", flush=True)
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
total = len(series) * len(CUTOFFS)
done = 0
t_start_all = time.time()

for ticker, cl in series.items():
    for cutoff in CUTOFFS:
        done += 1
        train = cl[(cl.index > cutoff - WINDOW) & (cl.index <= cutoff)]
        if len(train) < 60:
            continue
        df_fit = pd.DataFrame({"ds": train.index, "y": train.values})

        try:
            m = Prophet(
                interval_width=0.95,
                seasonality_mode=WINNING_COMBO["seasonality_mode"],
                growth="linear",
                changepoint_prior_scale=WINNING_COMBO["changepoint_prior_scale"],
                daily_seasonality=False,
                yearly_seasonality=WINNING_COMBO["yearly_seasonality"],
                weekly_seasonality=False,
            )
            m.fit(df_fit)
            future = m.make_future_dataframe(periods=FORECAST_PERIODS)
            pred = m.predict(future)[["ds", "yhat"]].set_index("ds")
        except Exception as e:
            print(f"fit error {ticker} {cutoff.date()}: {str(e)[:80]}", flush=True)
            continue

        future_actual = cl[(cl.index > cutoff) & (cl.index <= cutoff + pd.Timedelta(days=ACTUAL_WINDOW_DAYS))]
        if future_actual.empty:
            continue

        errs_all = []
        errs_by_date = []
        for d, actual_px in future_actual.items():
            if d not in pred.index:
                continue
            yhat_v = pred.loc[d, "yhat"]
            if yhat_v <= 0:
                continue
            e = (actual_px / yhat_v - 1) * 100
            errs_all.append(e)
            errs_by_date.append((d, e))

        if not errs_all:
            continue

        mape_0_30 = float(np.abs(errs_all).mean())
        row = {"ticker": ticker, "cutoff": cutoff.strftime("%Y-%m-%d"), "mape_0_30d": mape_0_30}

        errs_df = pd.DataFrame(errs_by_date, columns=["ds", "pct_err"])
        for cp in CHECKPOINTS:
            target_date = cutoff + pd.Timedelta(days=cp)
            nearest_idx = (errs_df["ds"] - target_date).abs().idxmin()
            nearest_date = errs_df.loc[nearest_idx, "ds"]
            if abs((nearest_date - target_date).days) > 4:
                row[f"abs_err_{cp}d"] = None
            else:
                row[f"abs_err_{cp}d"] = abs(float(errs_df.loc[nearest_idx, "pct_err"]))

        results.append(row)

    elapsed = time.time() - t_start_all
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    print(f"[{done}/{total}] ticker={ticker} completado -- {elapsed:.0f}s transcurridos, ~{remaining:.0f}s restantes", flush=True)

df_results = pd.DataFrame(results)
out_path = r"results\hp_validate_results.csv"
df_results.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path}  ({len(df_results)} filas)", flush=True)
print(f"Tiempo total: {time.time()-t_start_all:.0f}s", flush=True)
