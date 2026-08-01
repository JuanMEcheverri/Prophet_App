"""
Simulacion de portafolio con capital limitado y reinversion, dia por dia,
sobre 1 año de historia real. Ahora con SCREENING COMPLETO (igual que la
app real) + grid search sobre max_dev_pct y min_r2.

Escenarios de estrategia (fijos, ya ganadores de backtests anteriores):
  - Optimo:       sl_pct=0.15, tp_pct=0.20
  - Conservador:  sl_pct=0.10, tp_pct=0.20
  - max_hold=15, entry_dev=0.03 (iguales en ambos)

Screening (igual que Prophet_App/app.py):
  - R² >= min_r2                    -- GRID: [0.60, 0.68, 0.78]
  - sigma <= max_sigma_pct          -- fijo en default app (6%)
  - min_growth <= growth <= max_growth -- fijo en default app (2%-35%)
  - ADR <= max_adr_pct              -- fijo en default app (4%)
  - deviation entre -max_dev y -entry_dev (si es mas negativo que -max_dev,
    es "MUY BAJO", NO se compra)    -- GRID: [10%, 12%, 15%, 20%]

Los 5 checks de screening + max_dev salen del MISMO ajuste de Prophet que ya
se hace para yhat (R²/sigma/growth) mas High/Low descargado aparte (ADR) --
no cuesta ajustes extra. El grid de max_dev/min_r2 (4x3=12) se cruza con los
2 escenarios de SL/TP x 2 de slots = 48 simulaciones de portafolio, todas
compartiendo la MISMA cache de ajustes de Prophet (barato reevaluar distintos
umbrales de screening sobre el mismo fit).
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

# Screening fijo (defaults de la app, no se barren en el grid)
MAX_SIGMA_PCT  = 0.06
MIN_GROWTH_PCT = 0.02
MAX_GROWTH_PCT = 0.35
MAX_ADR_PCT    = 0.04

# Grid de screening que SI se barre
MAX_DEV_GRID = [0.10, 0.12, 0.15, 0.20]
MIN_R2_GRID  = [0.60, 0.68, 0.78]

SIM_DAYS_BACK = 365
SIM_START = TODAY - pd.Timedelta(days=SIM_DAYS_BACK)

SCENARIOS_SLTP = [("Optimo", 0.15, 0.20), ("Conservador", 0.10, 0.20)]
MAX_POSITIONS = [3, 5]

DOWNLOAD_START = (SIM_START - WINDOW - pd.Timedelta(days=100)).strftime("%Y-%m-%d")
DOWNLOAD_END = TODAY.strftime("%Y-%m-%d")

print(f"Tickers: {len(TICKERS)}  Simulacion desde: {SIM_START.date()} hasta {TODAY.date()}", flush=True)
print(f"Grid screening: max_dev={MAX_DEV_GRID}  min_r2={MIN_R2_GRID}  "
      f"({len(MAX_DEV_GRID)*len(MIN_R2_GRID)} combos) x {len(SCENARIOS_SLTP)*len(MAX_POSITIONS)} escenarios SL/TP/slots "
      f"= {len(MAX_DEV_GRID)*len(MIN_R2_GRID)*len(SCENARIOS_SLTP)*len(MAX_POSITIONS)} simulaciones de portafolio", flush=True)
print(f"Descarga: {DOWNLOAD_START} -> {DOWNLOAD_END}", flush=True)

t0 = time.time()
raw = yf.download(TICKERS, start=DOWNLOAD_START, end=DOWNLOAD_END, auto_adjust=True, progress=False)
print(f"Descarga completa en {time.time()-t0:.1f}s", flush=True)

cl_dict, hi_dict, lo_dict = {}, {}, {}
for t in TICKERS:
    try:
        cl = raw["Close"][t].dropna()
        hi = raw["High"][t].dropna()
        lo = raw["Low"][t].dropna()
        if cl.index.tz is not None:
            cl.index = cl.index.tz_localize(None)
            hi.index = hi.index.tz_localize(None)
            lo.index = lo.index.tz_localize(None)
        cl.index = cl.index.normalize(); hi.index = hi.index.normalize(); lo.index = lo.index.normalize()
        cl_dict[t] = cl; hi_dict[t] = hi; lo_dict[t] = lo
    except Exception as e:
        print(f"skip {t}: {e}", flush=True)

TICKERS = [t for t in TICKERS if t in cl_dict]

all_dates = sorted(set().union(*[set(cl_dict[t].index) for t in TICKERS]))
SIM_DATES = [d for d in all_dates if SIM_START <= d <= TODAY]
print(f"Dias de mercado a simular: {len(SIM_DATES)}", flush=True)

# ---- Cache de ajustes de Prophet + metricas de screening, compartida ----
# ticker -> {fit_date: (yhat_series, r2, sigma, growth, adr)}
# Se guardan VARIOS ajustes por ticker (no solo el ultimo) para que los 48
# combos del grid, que recorren las mismas fechas cada uno, puedan reusar
# ajustes hechos por un combo anterior en vez de re-entrenar desde cero.
fit_cache = {}
n_fits = [0]


def get_fit_info(ticker, today):
    ticker_cache = fit_cache.setdefault(ticker, {})
    best_fit_date, best_info = None, None
    for fit_date, info in ticker_cache.items():
        if fit_date <= today and (today - fit_date).days <= CACHE_VALID_DAYS and today in info[0].index:
            if best_fit_date is None or fit_date > best_fit_date:
                best_fit_date, best_info = fit_date, info
    if best_info is not None:
        return best_info

    cl, hi, lo = cl_dict[ticker], hi_dict[ticker], lo_dict[ticker]
    train = cl[(cl.index > today - WINDOW) & (cl.index <= today)]
    if len(train) < 60:
        return None

    tail_n = min(90, len(train))
    hi_tail = hi[hi.index.isin(train.index)].tail(tail_n)
    lo_tail = lo[lo.index.isin(train.index)].tail(tail_n)
    cl_tail = train.tail(tail_n)
    adr = float(((hi_tail - lo_tail) / cl_tail.clip(lower=0.01)).mean())

    df_fit = pd.DataFrame({"ds": train.index, "y": train.values})
    try:
        m = Prophet(**PROPHET_KW)
        m.fit(df_fit)
        future = m.make_future_dataframe(periods=FORECAST_PERIODS)
        pred_full = m.predict(future)[["ds", "yhat"]].set_index("ds")
    except Exception:
        return None
    n_fits[0] += 1
    yhat_series = pred_full["yhat"]

    # R2 / sigma in-sample (igual que train_models() en app.py)
    df_idx = df_fit.set_index("ds")
    common = [d for d in df_idx.index if d in yhat_series.index]
    y_is = np.array([df_idx.loc[d, "y"] for d in common])
    yh_is = np.array([yhat_series.loc[d] for d in common])
    resids = (y_is - yh_is) / np.clip(yh_is, 0.01, None)
    r2 = float(np.corrcoef(y_is, yh_is)[0, 1] ** 2) if len(common) > 1 else 0.0
    sigma = float(np.std(resids))

    # Growth anualizado sobre TODO el rango del ajuste (entrenamiento + proyeccion)
    yhat_sorted = yhat_series.sort_index()
    y0, y1 = float(yhat_sorted.iloc[0]), float(yhat_sorted.iloc[-1])
    nyrs = (yhat_sorted.index[-1] - yhat_sorted.index[0]).days / 365.25
    growth = (y1 / y0) ** (1 / nyrs) - 1 if nyrs > 0 and y0 > 0 else 0.0

    info = (yhat_series, r2, sigma, growth, adr)
    ticker_cache[today] = info
    return info


def passes_screening(r2, sigma, growth, adr, min_r2):
    return (r2 >= min_r2 and sigma <= MAX_SIGMA_PCT
            and MIN_GROWTH_PCT <= growth <= MAX_GROWTH_PCT
            and adr <= MAX_ADR_PCT)


all_transactions = []
all_summaries = []
t_sim_start = time.time()
n_combo_done = 0
n_combo_total = len(MAX_DEV_GRID) * len(MIN_R2_GRID) * len(SCENARIOS_SLTP) * len(MAX_POSITIONS)

for max_dev_pct in MAX_DEV_GRID:
    for min_r2 in MIN_R2_GRID:
        for scen_name, sl_pct, tp_pct in SCENARIOS_SLTP:
            for max_pos in MAX_POSITIONS:
                n_combo_done += 1
                slot_cap0 = INITIAL_CAPITAL / max_pos
                slots = [{"ticker": None, "cash": slot_cap0, "entry_price": None,
                          "entry_date": None, "shares": None, "entry_yhat": None,
                          "entry_dev_pct": None} for _ in range(max_pos)]
                held = set()

                for today in SIM_DATES:
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
                                "max_dev_pct": max_dev_pct, "min_r2": min_r2,
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

                    # 2) buscar entradas para slots libres (con screening completo)
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
                            info = get_fit_info(t, today)
                            if info is None:
                                continue
                            yhat_series, r2, sigma, growth, adr = info
                            if today not in yhat_series.index:
                                continue
                            yhat_v = float(yhat_series.loc[today])
                            if yhat_v <= 0:
                                continue
                            if not passes_screening(r2, sigma, growth, adr, min_r2):
                                continue
                            dev_pct = (px / yhat_v - 1) * 100
                            # zona de compra: entre -max_dev y -entry_dev (mas alla de
                            # -max_dev es "MUY BAJO", no se compra -- igual que classify() en app.py)
                            if -max_dev_pct * 100 < dev_pct <= -ENTRY_DEV * 100:
                                candidates.append((t, px, yhat_v, dev_pct))
                        candidates.sort(key=lambda c: c[3])

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
                                "max_dev_pct": max_dev_pct, "min_r2": min_r2,
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

                final_value = 0.0
                for i, slot in enumerate(slots):
                    if slot["ticker"] is not None:
                        cl = cl_dict[slot["ticker"]]
                        last_px = float(cl[cl.index <= TODAY].iloc[-1])
                        mark_value = slot["shares"] * last_px
                        final_value += mark_value
                        all_transactions.append({
                            "max_dev_pct": max_dev_pct, "min_r2": min_r2,
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

                n_tx = sum(1 for tr in all_transactions if tr["max_dev_pct"] == max_dev_pct and tr["min_r2"] == min_r2
                           and tr["escenario"] == scen_name and tr["max_slots"] == max_pos and tr["accion"] in ("COMPRA", "VENTA"))
                n_sells = sum(1 for tr in all_transactions if tr["max_dev_pct"] == max_dev_pct and tr["min_r2"] == min_r2
                              and tr["escenario"] == scen_name and tr["max_slots"] == max_pos and tr["accion"] == "VENTA")
                n_wins = sum(1 for tr in all_transactions if tr["max_dev_pct"] == max_dev_pct and tr["min_r2"] == min_r2
                             and tr["escenario"] == scen_name and tr["max_slots"] == max_pos and tr["accion"] == "VENTA" and tr["pnl_pct"] > 0)
                total_fees = n_tx * FEE

                all_summaries.append({
                    "max_dev_pct": max_dev_pct, "min_r2": min_r2,
                    "escenario": scen_name, "max_slots": max_pos,
                    "capital_inicial": INITIAL_CAPITAL,
                    "capital_final": round(final_value, 2),
                    "retorno_pct": round((final_value / INITIAL_CAPITAL - 1) * 100, 2),
                    "n_transacciones": n_tx,
                    "n_ventas_cerradas": n_sells,
                    "win_rate_pct": round(100 * n_wins / n_sells, 1) if n_sells else None,
                    "fees_totales": round(total_fees, 2),
                })

                elapsed = time.time() - t_sim_start
                print(f"[{n_combo_done}/{n_combo_total}] max_dev={max_dev_pct} min_r2={min_r2} "
                      f"{scen_name} x{max_pos}: capital final={final_value:.2f} "
                      f"({(final_value/INITIAL_CAPITAL-1)*100:+.1f}%) -- {n_tx} tx -- "
                      f"{elapsed:.0f}s transcurridos -- {n_fits[0]} ajustes Prophet", flush=True)

df_tx = pd.DataFrame(all_transactions)
df_summary = pd.DataFrame(all_summaries)
tx_path = r"results\portfolio2_transactions.csv"
sum_path = r"results\portfolio2_summary.csv"
df_tx.to_csv(tx_path, index=False)
df_summary.to_csv(sum_path, index=False)

print(f"\nGuardado transacciones: {tx_path} ({len(df_tx)} filas)", flush=True)
print(f"Guardado resumen: {sum_path} ({len(df_summary)} combos)", flush=True)
print(f"Total ajustes de Prophet (cache compartida entre TODOS los combos): {n_fits[0]}", flush=True)
print(f"Tiempo total: {time.time()-t_sim_start:.0f}s", flush=True)
