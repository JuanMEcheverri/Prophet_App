"""
Simulacion de portafolio con capital limitado y reinversion, dia por dia,
sobre 1 año de historia real. 4 escenarios: {Optimo, Conservador} x {3, 5 slots}.

- Optimo:       sl_pct=0.15, tp_pct=0.20
- Conservador:  sl_pct=0.10, tp_pct=0.20
- max_hold=15 dias (fijo, ganador de los backtests anteriores) y entry_dev=0.03
  (fijo) son iguales en ambos escenarios.
- Comision: 1 EUR por transaccion (compra o venta).
- Cuando un slot queda libre, se buscan candidatos entre los 55 tickers NO
  tenidos ya por otro slot de ESE escenario. Un ticker con senal de entrada
  se le asigna al slot con MAYOR desviacion bajo yhat si hay mas de un slot
  libre ese dia y solo una senal, se llena un solo slot y el otro espera al
  dia siguiente (nunca se duplica el mismo ticker en 2 slots a la vez).
- El ajuste de Prophet para cada ticker se cachea: valido mientras el dia
  consultado caiga dentro de los ~35 dias siguientes al ultimo ajuste. Si no,
  se re-entrena con datos hasta ese dia exacto (lo mas cercano a una
  simulacion real: se recalcula justo cuando hace falta, no en un calendario
  fijo). La cache se comparte entre los 4 escenarios.
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
    "ADBE","ADI","AEP","AMAT","AMD","AMGN","AMZN","AVGO","CDNS","CEG","COST",
    "CSCO","CTAS","DDOG","DLTR","DXCM","EXC","FAST","FSLR","FTNT","GEHC","GFS",
    "GILD","GOOG","GOOGL","IDXX","ISRG","KHC","KLAC","LRCX","MAR","MELI","META",
    "MNST","MRVL","MU","NTAP","NVDA","NXPI","ON","PANW","PAYX","PCAR","PDD",
    "QCOM","REGN","SNPS","TSLA","TTD","TXN","VRTX","XEL","ZBRA","ZM","ZS",
]
TICKERS = sorted(set(TICKERS))

WINDOW = pd.DateOffset(years=1)
PROPHET_KW = dict(
    interval_width=0.95, seasonality_mode="multiplicative", growth="linear",
    changepoint_prior_scale=0.03, daily_seasonality=False,
    yearly_seasonality=True, weekly_seasonality=False,
)

ENTRY_DEV = 0.03
MAX_HOLD  = 15
FEE       = 1.0
INITIAL_CAPITAL = 1000.0
FORECAST_PERIODS = 35
CACHE_VALID_DAYS = 35

SIM_DAYS_BACK = 365
SIM_START = TODAY - pd.Timedelta(days=SIM_DAYS_BACK)

SCENARIOS_SLTP = [("Optimo", 0.15, 0.20), ("Conservador", 0.10, 0.20)]
MAX_POSITIONS = [3, 5]

DOWNLOAD_START = (SIM_START - WINDOW - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
DOWNLOAD_END = TODAY.strftime("%Y-%m-%d")

print(f"Tickers: {len(TICKERS)}  Simulacion desde: {SIM_START.date()} hasta {TODAY.date()}", flush=True)
print(f"Descarga: {DOWNLOAD_START} -> {DOWNLOAD_END}", flush=True)

t0 = time.time()
raw = yf.download(TICKERS, start=DOWNLOAD_START, end=DOWNLOAD_END, auto_adjust=True, progress=False)
print(f"Descarga completa en {time.time()-t0:.1f}s", flush=True)

close_all = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": TICKERS[0]})

cl_dict = {}
for t in TICKERS:
    try:
        cl = close_all[t].dropna()
        if cl.index.tz is not None:
            cl.index = cl.index.tz_localize(None)
        cl.index = cl.index.normalize()
        cl_dict[t] = cl
    except Exception as e:
        print(f"skip {t}: {e}", flush=True)

TICKERS = [t for t in TICKERS if t in cl_dict]

# Calendario de simulacion: dias de mercado dentro del rango, usando la union
# de fechas de todos los tickers (deberian coincidir casi siempre, son todas
# acciones de EEUU).
all_dates = sorted(set().union(*[set(cl_dict[t].index) for t in TICKERS]))
SIM_DATES = [d for d in all_dates if SIM_START <= d <= TODAY]
print(f"Dias de mercado a simular: {len(SIM_DATES)}", flush=True)

# ---- Cache de ajustes de Prophet, compartida entre los 4 escenarios ----
fit_cache = {}   # ticker -> (fit_date, yhat_series pd.Series indexada por fecha)
n_fits = [0]


def get_yhat(ticker, today):
    entry = fit_cache.get(ticker)
    if entry is not None:
        fit_date, yhat_series = entry
        # CRITICO: fit_date <= today evita usar un ajuste hecho con datos del
        # "futuro" respecto al dia simulado (data leakage). Prophet.predict()
        # devuelve yhat para TODO el rango de entrenamiento + proyeccion, asi
        # que sin este chequeo un ajuste posterior "se cuela" para dias
        # anteriores que cayeron dentro de su ventana de entrenamiento.
        if fit_date <= today and today in yhat_series.index and (today - fit_date).days <= CACHE_VALID_DAYS:
            return float(yhat_series.loc[today])

    cl = cl_dict[ticker]
    train = cl[(cl.index > today - WINDOW) & (cl.index <= today)]
    if len(train) < 60:
        return None
    df_fit = pd.DataFrame({"ds": train.index, "y": train.values})
    try:
        m = Prophet(**PROPHET_KW)
        m.fit(df_fit)
        future = m.make_future_dataframe(periods=FORECAST_PERIODS)
        pred = m.predict(future)[["ds", "yhat"]].set_index("ds")["yhat"]
    except Exception:
        return None
    n_fits[0] += 1
    fit_cache[ticker] = (today, pred)
    if today in pred.index:
        return float(pred.loc[today])
    return None


all_transactions = []
all_summaries = []
t_sim_start = time.time()

for scen_name, sl_pct, tp_pct in SCENARIOS_SLTP:
    for max_pos in MAX_POSITIONS:
        slot_cap0 = INITIAL_CAPITAL / max_pos
        slots = [{"ticker": None, "cash": slot_cap0, "entry_price": None,
                  "entry_date": None, "shares": None, "entry_yhat": None,
                  "entry_dev_pct": None} for _ in range(max_pos)]
        held = set()
        n_days_done = 0

        for today in SIM_DATES:
            n_days_done += 1
            # 1) revisar salidas
            for i, slot in enumerate(slots):
                if slot["ticker"] is None:
                    continue
                cl = cl_dict[slot["ticker"]]
                if today not in cl.index:
                    continue
                px = float(cl.loc[today])
                days_since = (today - slot["entry_date"]).days
                sl_price = slot["entry_price"] * (1 - sl_pct)
                tp_price = slot["entry_price"] * (1 + tp_pct)
                exit_reason = None
                if px <= sl_price:
                    exit_reason = "SL"
                elif px >= tp_price:
                    exit_reason = "TP"
                elif days_since >= MAX_HOLD:
                    exit_reason = "MAX_HOLD"
                if exit_reason:
                    proceeds = slot["shares"] * px - FEE
                    pnl_pct = (px / slot["entry_price"] - 1) * 100
                    all_transactions.append({
                        "escenario": scen_name, "max_slots": max_pos, "slot": i,
                        "accion": "VENTA", "ticker": slot["ticker"],
                        "fecha_entrada": slot["entry_date"].strftime("%Y-%m-%d"),
                        "precio_entrada": round(slot["entry_price"], 2),
                        "yhat_entrada": round(slot["entry_yhat"], 2),
                        "dev_entrada_pct": round(slot["entry_dev_pct"], 2),
                        "fecha_salida": today.strftime("%Y-%m-%d"),
                        "precio_salida": round(px, 2),
                        "razon_salida": exit_reason,
                        "dias_en_posicion": days_since,
                        "pnl_pct": round(pnl_pct, 2),
                        "capital_slot_tras_venta": round(proceeds, 2),
                        "fee": FEE,
                    })
                    held.discard(slot["ticker"])
                    slot.update({"ticker": None, "cash": proceeds, "entry_price": None,
                                 "entry_date": None, "shares": None, "entry_yhat": None,
                                 "entry_dev_pct": None})

            # 2) buscar entradas para slots libres
            free_idxs = [i for i, s in enumerate(slots) if s["ticker"] is None]
            if free_idxs:
                candidates = []
                for t in TICKERS:
                    if t in held:
                        continue
                    cl = cl_dict[t]
                    if today not in cl.index:
                        continue
                    px = float(cl.loc[today])
                    yhat_v = get_yhat(t, today)
                    if yhat_v is None or yhat_v <= 0:
                        continue
                    dev_pct = (px / yhat_v - 1) * 100
                    if px < yhat_v * (1 - ENTRY_DEV):
                        candidates.append((t, px, yhat_v, dev_pct))
                candidates.sort(key=lambda c: c[3])  # mas negativo (mas sobrevendido) primero

                for i in free_idxs:
                    if not candidates:
                        break
                    t, px, yhat_v, dev_pct = candidates.pop(0)
                    slot = slots[i]
                    invest_cash = slot["cash"] - FEE
                    if invest_cash <= 0:
                        continue
                    shares = invest_cash / px
                    all_transactions.append({
                        "escenario": scen_name, "max_slots": max_pos, "slot": i,
                        "accion": "COMPRA", "ticker": t,
                        "fecha_entrada": today.strftime("%Y-%m-%d"),
                        "precio_entrada": round(px, 2),
                        "yhat_entrada": round(yhat_v, 2),
                        "dev_entrada_pct": round(dev_pct, 2),
                        "fecha_salida": None, "precio_salida": None,
                        "razon_salida": None, "dias_en_posicion": None,
                        "pnl_pct": None,
                        "capital_slot_tras_venta": None,
                        "fee": FEE,
                    })
                    slot.update({"ticker": t, "entry_price": px, "entry_date": today,
                                 "shares": shares, "entry_yhat": yhat_v, "entry_dev_pct": dev_pct})
                    held.add(t)

            if n_days_done % 50 == 0:
                elapsed = time.time() - t_sim_start
                print(f"[{scen_name} x{max_pos}] dia {n_days_done}/{len(SIM_DATES)} "
                      f"({today.date()}) -- {elapsed:.0f}s transcurridos -- "
                      f"{n_fits[0]} ajustes de Prophet hasta ahora (cache compartida)", flush=True)

        # cierre final: valorar posiciones abiertas al ultimo precio disponible
        final_value = 0.0
        for i, slot in enumerate(slots):
            if slot["ticker"] is not None:
                cl = cl_dict[slot["ticker"]]
                last_px = float(cl[cl.index <= TODAY].iloc[-1])
                mark_value = slot["shares"] * last_px
                final_value += mark_value
                all_transactions.append({
                    "escenario": scen_name, "max_slots": max_pos, "slot": i,
                    "accion": "ABIERTA_AL_CIERRE", "ticker": slot["ticker"],
                    "fecha_entrada": slot["entry_date"].strftime("%Y-%m-%d"),
                    "precio_entrada": round(slot["entry_price"], 2),
                    "yhat_entrada": round(slot["entry_yhat"], 2),
                    "dev_entrada_pct": round(slot["entry_dev_pct"], 2),
                    "fecha_salida": TODAY.strftime("%Y-%m-%d"),
                    "precio_salida": round(last_px, 2),
                    "razon_salida": "ABIERTA_AL_CIERRE",
                    "dias_en_posicion": (TODAY - slot["entry_date"]).days,
                    "pnl_pct": round((last_px / slot["entry_price"] - 1) * 100, 2),
                    "capital_slot_tras_venta": round(mark_value, 2),
                    "fee": 0.0,
                })
            else:
                final_value += slot["cash"]

        n_tx = sum(1 for tr in all_transactions if tr["escenario"] == scen_name and tr["max_slots"] == max_pos and tr["accion"] in ("COMPRA", "VENTA"))
        total_fees = n_tx * FEE
        n_sells = sum(1 for tr in all_transactions if tr["escenario"] == scen_name and tr["max_slots"] == max_pos and tr["accion"] == "VENTA")
        n_wins = sum(1 for tr in all_transactions if tr["escenario"] == scen_name and tr["max_slots"] == max_pos and tr["accion"] == "VENTA" and tr["pnl_pct"] > 0)

        all_summaries.append({
            "escenario": scen_name, "max_slots": max_pos,
            "capital_inicial": INITIAL_CAPITAL,
            "capital_final": round(final_value, 2),
            "retorno_pct": round((final_value / INITIAL_CAPITAL - 1) * 100, 2),
            "n_transacciones": n_tx,
            "n_ventas_cerradas": n_sells,
            "win_rate_pct": round(100 * n_wins / n_sells, 1) if n_sells else None,
            "fees_totales": round(total_fees, 2),
        })
        print(f"=== {scen_name} x{max_pos} slots: capital final={final_value:.2f} "
              f"({(final_value/INITIAL_CAPITAL-1)*100:+.1f}%) -- {n_tx} transacciones ===", flush=True)

df_tx = pd.DataFrame(all_transactions)
df_summary = pd.DataFrame(all_summaries)
tx_path = r"results\portfolio_transactions.csv"
sum_path = r"results\portfolio_summary.csv"
df_tx.to_csv(tx_path, index=False)
df_summary.to_csv(sum_path, index=False)

print(f"\nGuardado transacciones: {tx_path} ({len(df_tx)} filas)", flush=True)
print(f"Guardado resumen: {sum_path}", flush=True)
print(f"Total ajustes de Prophet (cache compartida entre 4 escenarios): {n_fits[0]}", flush=True)
print(f"Tiempo total: {time.time()-t_sim_start:.0f}s", flush=True)
