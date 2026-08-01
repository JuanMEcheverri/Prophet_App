"""
Backtest de la ESTRATEGIA COMPLETA (no solo precision de yhat):

Entrada: primer dia (dentro de ENTRY_SEARCH_DAYS desde el corte) en que el
         precio real cae por debajo de yhat * (1 - ENTRY_DEV) -- igual que
         la señal 🟢 COMPRAR de la app.

Salida (recorrido dia por dia, cronologico, desde la entrada):
  - Stop Loss:    precio real <= precio_entrada * (1 - SL_PCT)
  - Take Profit:  precio real >= precio_entrada * (1 + TP_PCT)
  - Vencimiento:  si pasan MAX_HOLD dias calendario sin TP ni SL

Se registra cada operacion simulada (entrada, salida, motivo, dias, P&L%)
para poder calcular win-rate / P&L promedio por ventana de entrenamiento
y por accion (para identificar cuales tickers conviene excluir).
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

# Nasdaq-100 completo (mismo universo que Prophet_App/app.py)
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

WINDOWS = {
    "6 meses": pd.DateOffset(months=6),
    "1 año":   pd.DateOffset(years=1),
    "2 años":  pd.DateOffset(years=2),
    "3 años":  pd.DateOffset(years=3),
    "4 años":  pd.DateOffset(years=4),
}

# Parametros de estrategia (defaults de la app, con el ajuste pedido por el usuario)
ENTRY_DEV = 0.03   # comprar si precio < yhat * (1 - 3%)
SL_PCT    = 0.03   # stop loss: -3% desde el precio de ENTRADA (antes 8%)
TP_PCT    = 0.05   # take profit: +5% desde el precio de ENTRADA (antes sobre yhat)
MAX_HOLD  = 30     # dias calendario maximos en posicion

ENTRY_SEARCH_DAYS = 35   # ventana para buscar una señal de entrada tras el corte
POST_ENTRY_BUFFER = 40   # dias extra de datos reales despues de la entrada mas tardia posible

N_CUTOFFS = 10
STEP_DAYS = 55
last_cutoff = TODAY - pd.Timedelta(days=ENTRY_SEARCH_DAYS + POST_ENTRY_BUFFER)
CUTOFFS = [last_cutoff - pd.Timedelta(days=STEP_DAYS * i) for i in range(N_CUTOFFS)]

DOWNLOAD_START = min(c - w for c in CUTOFFS for w in WINDOWS.values()).strftime("%Y-%m-%d")
DOWNLOAD_END = TODAY.strftime("%Y-%m-%d")

print(f"Tickers: {len(TICKERS)}", flush=True)
print(f"Cutoffs: {[c.strftime('%Y-%m-%d') for c in CUTOFFS]}", flush=True)
print(f"Estrategia: entry_dev={ENTRY_DEV} sl_pct={SL_PCT} tp_pct={TP_PCT} max_hold={MAX_HOLD}", flush=True)
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
total = len(series) * len(WINDOWS) * len(CUTOFFS)
done = 0
t_start_all = time.time()

for ticker, cl in series.items():
    for cutoff in CUTOFFS:
        for wname, wdelta in WINDOWS.items():
            done += 1
            train_start = cutoff - wdelta
            train = cl[(cl.index > train_start) & (cl.index <= cutoff)]
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
                future = m.make_future_dataframe(periods=ENTRY_SEARCH_DAYS)
                pred = m.predict(future)[["ds", "yhat"]].set_index("ds")
            except Exception as e:
                print(f"fit error {ticker} {cutoff.date()} {wname}: {str(e)[:80]}", flush=True)
                continue

            # 1) buscar señal de entrada dentro de ENTRY_SEARCH_DAYS
            search_actual = cl[(cl.index > cutoff) & (cl.index <= cutoff + pd.Timedelta(days=ENTRY_SEARCH_DAYS))]
            entry_date = entry_price = None
            for d, px in search_actual.items():
                if d not in pred.index:
                    continue
                yhat_v = pred.loc[d, "yhat"]
                if yhat_v <= 0:
                    continue
                if px < yhat_v * (1 - ENTRY_DEV):
                    entry_date, entry_price = d, float(px)
                    break

            if entry_date is None:
                continue  # sin señal de compra en este ciclo

            # 2) simular la salida dia por dia (SL / TP fijos desde el precio de entrada)
            sl_price = entry_price * (1 - SL_PCT)
            tp_price = entry_price * (1 + TP_PCT)
            hold_window = cl[(cl.index > entry_date) & (cl.index <= entry_date + pd.Timedelta(days=MAX_HOLD + 10))]

            exit_date = exit_price = exit_reason = None
            for d, px in hold_window.items():
                days_since_entry = (d - entry_date).days
                px = float(px)
                if px <= sl_price:
                    exit_date, exit_price, exit_reason = d, px, "SL"
                    break
                if px >= tp_price:
                    exit_date, exit_price, exit_reason = d, px, "TP"
                    break
                if days_since_entry >= MAX_HOLD:
                    exit_date, exit_price, exit_reason = d, px, "MAX_HOLD"
                    break

            if exit_date is None:
                continue  # no hay suficientes datos futuros para cerrar la operacion

            pnl_pct = (exit_price / entry_price - 1) * 100
            trades.append({
                "ticker": ticker, "window": wname, "cutoff": cutoff.strftime("%Y-%m-%d"),
                "entry_date": entry_date.strftime("%Y-%m-%d"), "entry_price": entry_price,
                "exit_date": exit_date.strftime("%Y-%m-%d"), "exit_reason": exit_reason,
                "days_held": (exit_date - entry_date).days, "pnl_pct": pnl_pct,
            })

    elapsed = time.time() - t_start_all
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    print(f"[{done}/{total}] ticker={ticker} completado -- {elapsed:.0f}s transcurridos, ~{remaining:.0f}s restantes -- {len(trades)} trades hasta ahora", flush=True)

df_trades = pd.DataFrame(trades)
out_path = r"results\trades.csv"
df_trades.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path}  ({len(df_trades)} operaciones simuladas)", flush=True)
print(f"Tiempo total: {time.time()-t_start_all:.0f}s", flush=True)
