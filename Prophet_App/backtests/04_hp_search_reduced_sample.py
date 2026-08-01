"""
Fase de busqueda: cual combinacion de hiperparametros de Prophet predice
mejor el yhat, con la ventana ya fija en "1 año" (la ganadora del backtest
anterior). Muestra reducida (40 tickers x 6 cortes) para que el grid sea
manejable; el ganador se valida despues con el universo completo.

Metodologia identica al primer backtest (MAPE 0-30d + checkpoints), solo
que ahora variamos hiperparametros de Prophet en vez de la ventana.
"""
import time, warnings, logging, random
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet

TODAY = pd.Timestamp("2026-08-01")

ALL_TICKERS = [
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
ALL_TICKERS = sorted(set(ALL_TICKERS))
TICKERS = random.Random(42).sample(ALL_TICKERS, 40)
TICKERS = sorted(TICKERS)

WINDOW = pd.DateOffset(years=1)  # ganadora del backtest de ventanas

HP_GRID = {
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.3, 0.5],
    "seasonality_mode":        ["multiplicative", "additive"],
    "yearly_seasonality":      [True, False],
}
COMBOS = [
    {"changepoint_prior_scale": cps, "seasonality_mode": sm, "yearly_seasonality": ys}
    for cps in HP_GRID["changepoint_prior_scale"]
    for sm in HP_GRID["seasonality_mode"]
    for ys in HP_GRID["yearly_seasonality"]
]

FORECAST_PERIODS = 33
ACTUAL_WINDOW_DAYS = 33
CHECKPOINTS = [5, 10, 15, 20, 25, 30]

N_CUTOFFS = 6
STEP_DAYS = 55
last_cutoff = TODAY - pd.Timedelta(days=ACTUAL_WINDOW_DAYS)
CUTOFFS = [last_cutoff - pd.Timedelta(days=STEP_DAYS * i) for i in range(N_CUTOFFS)]

DOWNLOAD_START = min(c - WINDOW for c in CUTOFFS).strftime("%Y-%m-%d")
DOWNLOAD_END = TODAY.strftime("%Y-%m-%d")

print(f"Tickers muestra: {len(TICKERS)} de {len(ALL_TICKERS)}", flush=True)
print(f"Cutoffs: {[c.strftime('%Y-%m-%d') for c in CUTOFFS]}", flush=True)
print(f"Combos de hiperparametros: {len(COMBOS)}", flush=True)
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
total = len(series) * len(CUTOFFS) * len(COMBOS)
done = 0
t_start_all = time.time()

for ticker, cl in series.items():
    for cutoff in CUTOFFS:
        train = cl[(cl.index > cutoff - WINDOW) & (cl.index <= cutoff)]
        if len(train) < 60:
            done += len(COMBOS)
            continue
        df_fit = pd.DataFrame({"ds": train.index, "y": train.values})

        future_actual = cl[(cl.index > cutoff) & (cl.index <= cutoff + pd.Timedelta(days=ACTUAL_WINDOW_DAYS))]

        for combo in COMBOS:
            done += 1
            combo_key = f"cps={combo['changepoint_prior_scale']}|mode={combo['seasonality_mode']}|yearly={combo['yearly_seasonality']}"
            try:
                m = Prophet(
                    interval_width=0.95,
                    seasonality_mode=combo["seasonality_mode"],
                    growth="linear",
                    changepoint_prior_scale=combo["changepoint_prior_scale"],
                    daily_seasonality=False,
                    yearly_seasonality=combo["yearly_seasonality"],
                    weekly_seasonality=False,
                )
                m.fit(df_fit)
                future = m.make_future_dataframe(periods=FORECAST_PERIODS)
                pred = m.predict(future)[["ds", "yhat"]].set_index("ds")
            except Exception as e:
                print(f"fit error {ticker} {cutoff.date()} {combo_key}: {str(e)[:80]}", flush=True)
                continue

            if future_actual.empty:
                continue

            errs = []
            for d, actual_px in future_actual.items():
                if d not in pred.index:
                    continue
                yhat_v = pred.loc[d, "yhat"]
                if yhat_v <= 0:
                    continue
                errs.append((actual_px / yhat_v - 1) * 100)

            if not errs:
                continue

            errs = np.array(errs)
            mape_0_30 = float(np.abs(errs).mean())

            results.append({
                "ticker": ticker, "cutoff": cutoff.strftime("%Y-%m-%d"), "combo": combo_key,
                "changepoint_prior_scale": combo["changepoint_prior_scale"],
                "seasonality_mode": combo["seasonality_mode"],
                "yearly_seasonality": combo["yearly_seasonality"],
                "mape_0_30d": mape_0_30,
            })

    elapsed = time.time() - t_start_all
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    print(f"[{done}/{total}] ticker={ticker} completado -- {elapsed:.0f}s transcurridos, ~{remaining:.0f}s restantes", flush=True)

df_results = pd.DataFrame(results)
out_path = r"results\hp_search_results.csv"
df_results.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path}  ({len(df_results)} filas)", flush=True)
print(f"Tiempo total: {time.time()-t_start_all:.0f}s", flush=True)
