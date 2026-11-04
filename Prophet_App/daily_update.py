"""Paso diario de la cartera de papel, sin Streamlit (para la rutina en la nube o para correrlo a mano).

1. Descarga 14 meses de precios de los 87 tickers + QQQ.
2. Toma la ultima sesion COMPLETA (si corre durante la sesion de EE.UU., usa la anterior).
3. Ajusta Prophet por ticker con cierres hasta esa sesion (ventana de 1 año) -> yhat del dia + screening
   con los umbrales de screening.json.
4. Avanza las dos carteras de papel (paper_bt15.json, paper_combo.json) y escribe paper_page.html.
Uso: python daily_update.py            (desde cualquier directorio)
"""
import os, sys, json, time, warnings, logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "backtests"))

import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet

import paper
import paper_page
from wf15 import common as C

TRAIN_WINDOW = pd.DateOffset(years=1)
MIN_ROWS = 200
FUTURE_DAYS = 43


def load_screening():
    try:
        with open(os.path.join(APP_DIR, "screening.json"), encoding="utf-8") as f:
            s = json.load(f)
    except OSError:
        s = {"min_r2": 0.5, "max_sigma_pct": 15, "min_growth_pct": -50, "max_growth_pct": 20, "max_adr_pct": 4}
    return s, dict(min_r2=s["min_r2"], max_sigma=s["max_sigma_pct"] / 100, min_growth=s["min_growth_pct"] / 100,
                   max_growth=s["max_growth_pct"] / 100, max_adr=s["max_adr_pct"] / 100)


def app_version():
    try:
        import re
        txt = open(os.path.join(os.path.dirname(APP_DIR), "VERSIONES.md"), encoding="utf-8").read()
        m = re.search(r"\*\*(\d+\.\d+)\*\*", txt)
        return m.group(1) if m else "?"
    except OSError:
        return "?"


def last_complete_session(cal):
    now_et = datetime.now(ZoneInfo("America/New_York"))
    last = cal[-1]
    if last.date() < now_et.date() or now_et.time() >= dtime(16, 10):
        return last
    return cal[-2]


def main():
    t0 = time.time()
    scr_raw, scr = load_screening()
    tickers = C.TICKERS
    raw = yf.download(tickers + ["QQQ"], period="14mo", auto_adjust=True, progress=False)
    close = raw["Close"]; opn = raw["Open"]; high = raw["High"]; low = raw["Low"]
    for df in (close, opn, high, low):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
    cal = close["QQQ"].dropna().index
    D = last_complete_session(cal)
    i = cal.get_loc(D)
    print(f"[daily] ultima sesion completa: {D.date()}  (hoy ET {datetime.now(ZoneInfo('America/New_York')):%Y-%m-%d %H:%M})", flush=True)

    day_data, tick_data, n_fit = {}, {}, 0
    r2 = sigma = growth = adr = None
    for t in tickers:
        cl = close[t].dropna()
        cl = cl[cl.index <= D]
        if len(cl) == 0 or cl.index[-1] != D:
            continue
        train = cl[(cl.index > D - TRAIN_WINDOW) & (cl.index <= D)]
        yhat, screen, pred = None, False, None
        r2 = sigma = growth = adr = None
        if len(train) >= MIN_ROWS:
            try:
                m = Prophet(**C.PROPHET_KW)
                m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
                pred = m.predict(m.make_future_dataframe(periods=FUTURE_DAYS))[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                pred["ds"] = pred["ds"].dt.normalize()
                pred = pred.set_index("ds")
                yhat = float(pred["yhat"].get(D))
                r2, sigma, growth, adr = C.screening_metrics(train, high[t].reindex(train.index), low[t].reindex(train.index), pred)
                screen = C.passes_screen(r2, sigma, growth, adr, scr)
                n_fit += 1
            except Exception as e:
                print(f"  {t}: fallo el ajuste ({str(e)[:60]})", flush=True)
                pred = None
        o = opn[t].get(D)
        prev = cl.iloc[:-1]
        day_data[t] = dict(open=(None if o is None or pd.isna(o) else float(o)), close=float(cl.iloc[-1]),
                           prev_close=(float(prev.iloc[-1]) if len(prev) else None), yhat=yhat, screen=bool(screen))
        td = dict(close=round(float(cl.iloc[-1]), 4), prev=(round(float(prev.iloc[-1]), 4) if len(prev) else None), yhat=yhat,
                  r2=(None if r2 is None else round(r2, 4)), sigma=(None if sigma is None else round(sigma, 5)),
                  growth=(None if growth is None else round(growth, 4)), adr=(None if adr is None else round(adr, 5)),
                  hist_d=[d.strftime("%Y-%m-%d") for d in train.index], hist_y=[round(float(v), 2) for v in train.values])
        if pred is not None:
            td.update(fit_d=[d.strftime("%Y-%m-%d") for d in pred.index], yhat_s=[round(float(v), 2) for v in pred["yhat"]],
                      lo=[round(float(v), 2) for v in pred["yhat_lower"]], hi=[round(float(v), 2) for v in pred["yhat_upper"]])
        tick_data[t] = td
    print(f"[daily] {n_fit} ajustes Prophet, {sum(1 for d in day_data.values() if d['screen'])} pasan screening ({time.time()-t0:.0f}s)", flush=True)

    states = {}
    for key in paper.STRATEGIES:
        path = os.path.join(APP_DIR, f"paper_{key}.json")
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
        except OSError:
            state = paper.new_state(key)
        before = state["last_date"]
        state = paper.step(state, D.date(), day_data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        s = paper.summary(state)
        print(f"[daily] {key}: {'sesion ya procesada' if before == state['last_date'] else 'procesada ' + state['last_date']} · "
              f"valor {s['equity']:.2f} ({s['ret']:+.1%}) · cerradas {s['n_closed']}", flush=True)
        for ev in state["log"][-1]["events"] if state["log"] else []:
            print("   ", ev, flush=True)
        states[key] = state

    def _json(name, default):
        try:
            with open(os.path.join(APP_DIR, name), encoding="utf-8") as f:
                return json.load(f)
        except OSError:
            return default
    est = _json("estrategia.json", {"entry_dev_pct": 4, "exit_dev_pct": 7, "sl_pct_pct": 7, "max_hold": 60, "max_dev_pct": 15})
    try:
        versions_md = open(os.path.join(os.path.dirname(APP_DIR), "VERSIONES.md"), encoding="utf-8").read()
    except OSError:
        versions_md = ""
    page_data = dict(
        D=D.date().isoformat(), version=app_version(), tickers=tick_data,
        screening=dict(min_r2=scr_raw["min_r2"], max_sigma=scr_raw["max_sigma_pct"], min_growth=scr_raw["min_growth_pct"],
                       max_growth=scr_raw["max_growth_pct"], max_adr=scr_raw["max_adr_pct"]),
        strategy=dict(entry=est["entry_dev_pct"], exit=est["exit_dev_pct"], sl=est["sl_pct_pct"], hold=est["max_hold"], max_dev=est["max_dev_pct"]),
        positions=_json("positions.json", []), paper=states, cfgs=paper.STRATEGIES, versions_md=versions_md,
    )
    html = paper_page.build(page_data)
    out = os.path.join(APP_DIR, "paper_page.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[daily] pagina -> {out} ({len(html)//1024} KB) · total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
