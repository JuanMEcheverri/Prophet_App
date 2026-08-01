"""
Grid search de reglas de entrada/salida (entry_dev, sl_pct, tp_pct, max_hold),
con Prophet ya fijo (ventana=1 año, changepoint_prior_scale=0.03,
seasonality_mode=multiplicative -- los ganadores de los backtests anteriores).

Optimizacion clave: Prophet se entrena UNA sola vez por (ticker, corte); las
144 combinaciones de reglas se simulan sobre esa misma curva de yhat (barato,
sin refitear).
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
PROPHET_KW = dict(
    interval_width=0.95, seasonality_mode="multiplicative", growth="linear",
    changepoint_prior_scale=0.03, daily_seasonality=False,
    yearly_seasonality=True, weekly_seasonality=False,
)

ENTRY_DEVS = [0.02, 0.03, 0.04, 0.05]
SL_PCTS    = [0.02, 0.03, 0.05]
TP_PCTS    = [0.03, 0.05, 0.07, 0.10]
MAX_HOLDS  = [15, 20, 30]

ENTRY_SEARCH_DAYS = 35
MAX_MAX_HOLD = max(MAX_HOLDS)
POST_ENTRY_BUFFER = MAX_MAX_HOLD + 10

N_CUTOFFS = 10
STEP_DAYS = 55
last_cutoff = TODAY - pd.Timedelta(days=ENTRY_SEARCH_DAYS + POST_ENTRY_BUFFER)
CUTOFFS = [last_cutoff - pd.Timedelta(days=STEP_DAYS * i) for i in range(N_CUTOFFS)]

DOWNLOAD_START = min(c - WINDOW for c in CUTOFFS).strftime("%Y-%m-%d")
DOWNLOAD_END = TODAY.strftime("%Y-%m-%d")

n_combos = len(ENTRY_DEVS) * len(SL_PCTS) * len(TP_PCTS) * len(MAX_HOLDS)
print(f"Tickers: {len(TICKERS)}  Cutoffs: {len(CUTOFFS)}  Combos de reglas: {n_combos}", flush=True)
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

trades = []
total_fits = len(series) * len(CUTOFFS)
done_fits = 0
t_start_all = time.time()

for ticker, cl in series.items():
    for cutoff in CUTOFFS:
        done_fits += 1
        train = cl[(cl.index > cutoff - WINDOW) & (cl.index <= cutoff)]
        if len(train) < 60:
            continue
        df_fit = pd.DataFrame({"ds": train.index, "y": train.values})

        try:
            m = Prophet(**PROPHET_KW)
            m.fit(df_fit)
            future = m.make_future_dataframe(periods=ENTRY_SEARCH_DAYS)
            pred = m.predict(future)[["ds", "yhat"]].set_index("ds")
        except Exception as e:
            print(f"fit error {ticker} {cutoff.date()}: {str(e)[:80]}", flush=True)
            continue

        search_actual = cl[(cl.index > cutoff) & (cl.index <= cutoff + pd.Timedelta(days=ENTRY_SEARCH_DAYS))]
        if search_actual.empty:
            continue

        # Para cada entry_dev, encontrar el primer dia de entrada (barato, sin refitear)
        for entry_dev in ENTRY_DEVS:
            entry_date = entry_price = None
            for d, px in search_actual.items():
                if d not in pred.index:
                    continue
                yhat_v = pred.loc[d, "yhat"]
                if yhat_v <= 0:
                    continue
                if px < yhat_v * (1 - entry_dev):
                    entry_date, entry_price = d, float(px)
                    break
            if entry_date is None:
                continue

            hold_window = cl[(cl.index > entry_date) & (cl.index <= entry_date + pd.Timedelta(days=POST_ENTRY_BUFFER))]
            if hold_window.empty:
                continue
            hold_days = (hold_window.index - entry_date).days.values
            hold_px = hold_window.values.astype(float)

            for sl_pct in SL_PCTS:
                sl_price = entry_price * (1 - sl_pct)
                for tp_pct in TP_PCTS:
                    tp_price = entry_price * (1 + tp_pct)
                    for max_hold in MAX_HOLDS:
                        exit_price = exit_reason = exit_days = None
                        for dsince, px in zip(hold_days, hold_px):
                            if px <= sl_price:
                                exit_price, exit_reason, exit_days = px, "SL", int(dsince)
                                break
                            if px >= tp_price:
                                exit_price, exit_reason, exit_days = px, "TP", int(dsince)
                                break
                            if dsince >= max_hold:
                                exit_price, exit_reason, exit_days = px, "MAX_HOLD", int(dsince)
                                break
                        if exit_price is None:
                            continue
                        pnl_pct = (exit_price / entry_price - 1) * 100
                        trades.append({
                            "ticker": ticker, "cutoff": cutoff.strftime("%Y-%m-%d"),
                            "entry_dev": entry_dev, "sl_pct": sl_pct, "tp_pct": tp_pct, "max_hold": max_hold,
                            "days_held": exit_days, "exit_reason": exit_reason, "pnl_pct": pnl_pct,
                        })

    elapsed = time.time() - t_start_all
    rate = done_fits / elapsed if elapsed > 0 else 0
    remaining = (total_fits - done_fits) / rate if rate > 0 else 0
    print(f"[{done_fits}/{total_fits} fits] ticker={ticker} -- {elapsed:.0f}s transcurridos, "
          f"~{remaining:.0f}s restantes -- {len(trades)} trades simulados hasta ahora", flush=True)

df_trades = pd.DataFrame(trades)
out_path = r"results\rules_search_results.csv"
df_trades.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path}  ({len(df_trades)} operaciones simuladas)", flush=True)
print(f"Tiempo total: {time.time()-t_start_all:.0f}s", flush=True)
