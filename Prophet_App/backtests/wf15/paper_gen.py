"""Genera el historico de la cartera de papel dia a dia con el banco de ajustes DIARIO (refit1).

Para cada dia D del rango: yhat(D) viene del ajuste cuyo fit_date == D (entrenado con cierres <= D),
el screening de las metricas de ese mismo ajuste, y los precios de D (apertura, cierre, cierre previo).
Nada posterior a D entra en la decision de D; la apertura de D+1 solo se usa al llegar D+1.
"""
import os, sys, json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import paper                      # Prophet_App/paper.py
from . import common as C


def build_day_data(wide, cal, bank, meta, D):
    row_b = bank[(bank["fit_date"] == D) & (bank["ds"] == D)].set_index("ticker")["yhat"]
    row_m = meta[(meta["fit_date"] == D) & (meta["ok"])].set_index("ticker")
    out = {}
    for t in C.TICKERS:
        cl = wide["close"][t].dropna()
        close = cl.get(D)
        if close is None or pd.isna(close):
            continue
        op = wide["open"][t].get(D)
        prev = cl[cl.index < D]          # cierre anterior del propio ticker (no por posicion: el indice incluye dias sin bolsa)
        m = row_m.loc[t] if t in row_m.index else None
        out[t] = dict(
            open=(None if op is None or pd.isna(op) else float(op)), close=float(close),
            prev_close=(float(prev.iloc[-1]) if len(prev) else None),
            yhat=(float(row_b[t]) if t in row_b.index else None),
            screen=bool(m is not None and C.passes_screen(m["r2"], m["sigma"], m["growth"], m["adr"])),
        )
    return out


def run(start="2026-09-01", end=None, refit_every=1):
    from .fit_bank import load_bank
    wide = C.load_prices()
    cal = C.trading_calendar(wide)
    bank, meta = load_bank(refit_every)
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= (pd.Timestamp(end) if end else cal[-1]))]
    app_dir = os.path.dirname(C.BT_DIR)      # Prophet_App/
    for key in paper.STRATEGIES:
        state = paper.new_state(key)
        for D in days:
            state = paper.step(state, D.date(), build_day_data(wide, cal, bank, meta, D))
        path = os.path.join(app_dir, f"paper_{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        s = paper.summary(state)
        print(f"[paper] {key}: {len(days)} dias {days[0].date()} -> {days[-1].date()}  equity {s['equity']:.2f} "
              f"({s['ret']:+.1%})  cerradas {s['n_closed']}  abiertas {sum(1 for x in state['slots'] if x['position'])}  -> {path}", flush=True)
        for e in state["log"]:
            for ev in e["events"]:
                if "COMPRA" in ev or "VENTA" in ev:
                    print(f"   {e['date']}  {ev}")
