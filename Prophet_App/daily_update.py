"""Paso diario de la cartera de papel, sin Streamlit (para la rutina en la nube o para correrlo a mano).

1. Descarga 14 meses de precios de los 87 tickers + QQQ.
2. Toma la ultima sesion COMPLETA (si corre durante la sesion de EE.UU., usa la anterior).
3. Ajusta Prophet por ticker con cierres hasta esa sesion (ventana de 1 año) -> yhat del dia + screening
   con los umbrales de screening.json.
4. Avanza las dos carteras de papel (paper_bt15.json, paper_combo.json) y escribe paper_page.html.
Uso: python daily_update.py            (desde cualquier directorio)
"""
import os, sys, json, time, warnings, logging
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
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
    if len(cal) < 2:
        raise SystemExit("[daily] ERROR: no se pudieron descargar precios de Yahoo Finance (0 sesiones de QQQ). "
                         "Revisa la red del entorno: debe permitir *.yahoo.com y *.finance.yahoo.com")
    D = last_complete_session(cal)
    i = cal.get_loc(D)
    print(f"[daily] ultima sesion completa: {D.date()}  (hoy ET {datetime.now(ZoneInfo('America/New_York')):%Y-%m-%d %H:%M})", flush=True)

    def build_day(Dx, full):
        """day_data (y tick_data si full) de la sesion Dx, con ajustes Prophet entrenados solo con cierres <= Dx."""
        day_data, tick_data, n_fit = {}, {}, 0
        for t in tickers:
            cl = close[t].dropna()
            cl = cl[cl.index <= Dx]
            if len(cl) == 0 or cl.index[-1] != Dx:
                continue
            train = cl[(cl.index > Dx - TRAIN_WINDOW) & (cl.index <= Dx)]
            yhat, screen, pred = None, False, None
            r2 = sigma = growth = adr = None
            if len(train) >= MIN_ROWS:
                try:
                    m = Prophet(**C.PROPHET_KW)
                    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
                    pred = m.predict(m.make_future_dataframe(periods=FUTURE_DAYS))[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                    pred["ds"] = pred["ds"].dt.normalize()
                    pred = pred.set_index("ds")
                    yhat = float(pred["yhat"].get(Dx))
                    r2, sigma, growth, adr = C.screening_metrics(train, high[t].reindex(train.index), low[t].reindex(train.index), pred)
                    screen = C.passes_screen(r2, sigma, growth, adr, scr)
                    n_fit += 1
                except Exception as e:
                    print(f"  {t}: fallo el ajuste ({str(e)[:60]})", flush=True)
                    pred = None
            o = opn[t].get(Dx)
            prev = cl.iloc[:-1]
            day_data[t] = dict(open=(None if o is None or pd.isna(o) else float(o)), close=float(cl.iloc[-1]),
                               prev_close=(float(prev.iloc[-1]) if len(prev) else None), yhat=yhat, screen=bool(screen),
                               metrics=(None if r2 is None else dict(r2=float(r2), sigma=float(sigma), growth=float(growth), adr=float(adr))))
            if full:
                td = dict(close=round(float(cl.iloc[-1]), 4), prev=(round(float(prev.iloc[-1]), 4) if len(prev) else None), yhat=yhat,
                          r2=(None if r2 is None else round(r2, 4)), sigma=(None if sigma is None else round(sigma, 5)),
                          growth=(None if growth is None else round(growth, 4)), adr=(None if adr is None else round(adr, 5)),
                          hist_d=[d.strftime("%Y-%m-%d") for d in train.index], hist_y=[round(float(v), 2) for v in train.values])
                if pred is not None:
                    td.update(fit_d=[d.strftime("%Y-%m-%d") for d in pred.index], yhat_s=[round(float(v), 2) for v in pred["yhat"]],
                              lo=[round(float(v), 2) for v in pred["yhat_lower"]], hi=[round(float(v), 2) for v in pred["yhat_upper"]])
                tick_data[t] = td
        print(f"[daily] sesion {Dx.date()}: {n_fit} ajustes Prophet, {sum(1 for d in day_data.values() if d['screen'])} pasan screening ({time.time()-t0:.0f}s)", flush=True)
        return day_data, tick_data

    day_cache = {}
    def day_for(Dx):
        if Dx not in day_cache:
            day_cache[Dx] = build_day(Dx, full=(Dx == D))
        return day_cache[Dx]

    day_data, tick_data = day_for(D)

    states = {}
    for key in paper.STRATEGIES:
        path = os.path.join(APP_DIR, f"paper_{key}.json")
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
        except OSError:
            state = paper.new_state(key)
        before = state["last_date"]
        # rellenar sesiones perdidas (PC apagado, fallo) una a una, cada una con sus propios ajustes
        pending = [x for x in cal if x <= D and (before is None or x.date().isoformat() > before)]
        if before is None:
            pending = [D]
        if len(pending) > 1:
            print(f"[daily] {key}: rellenando {len(pending)} sesiones pendientes: {', '.join(str(x.date()) for x in pending)}", flush=True)
        for Dx in pending:
            state = paper.step(state, Dx.date(), day_for(Dx)[0])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        s = paper.summary(state)
        print(f"[daily] {key}: {'sesion ya procesada' if before == state['last_date'] else 'procesada hasta ' + state['last_date']} · "
              f"valor {s['equity']:.2f} ({s['ret']:+.1%}) · cerradas {s['n_closed']}", flush=True)
        for e in state["log"][-len(pending):] if pending and before != state["last_date"] else []:
            for ev in e["events"]:
                print("   ", e["date"], ev, flush=True)
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
    from names import NAMES
    import poslog
    positions = _json("positions.json", [])
    plog = _json("positions_log.json", [])
    added = False
    last_logged = max((e["date"] for e in plog if not e.get("parcial")), default=None)   # las parciales se rehacen
    first_pos = min((p["fecha_entrada"] for p in positions), default=None)
    for Dx in [x for x in cal if x <= D and (last_logged is None or x.date().isoformat() > last_logged)
               and (first_pos is not None and x.date().isoformat() >= first_pos)]:
        plog, a = poslog.update_log(plog, Dx.date().isoformat(), positions, day_for(Dx)[0],
                                    dict(sl=est["sl_pct_pct"], exit_dev=est["exit_dev_pct"], hold=est["max_hold"]))
        added = added or a
    if added:
        with open(os.path.join(APP_DIR, "positions_log.json"), "w", encoding="utf-8") as f:
            json.dump(plog, f, indent=2, ensure_ascii=False)
        print(f"[daily] diario de mis posiciones: sesion {D.date()} añadida ({len(plog[-1]['events'])} lineas)", flush=True)
    page_data = dict(
        D=D.date().isoformat(), version=app_version(), tickers=tick_data, names=NAMES,
        screening=dict(min_r2=scr_raw["min_r2"], max_sigma=scr_raw["max_sigma_pct"], min_growth=scr_raw["min_growth_pct"],
                       max_growth=scr_raw["max_growth_pct"], max_adr=scr_raw["max_adr_pct"]),
        strategy=dict(entry=est["entry_dev_pct"], exit=est["exit_dev_pct"], sl=est["sl_pct_pct"], hold=est["max_hold"], max_dev=est["max_dev_pct"]),
        positions=positions, positions_log=plog, paper=states, cfgs=paper.STRATEGIES, versions_md=versions_md,
    )
    write_slack_summary(D.date().isoformat(), tick_data, positions, est, states, scr_raw)
    html = paper_page.build(page_data)
    out = os.path.join(APP_DIR, "paper_page.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[daily] pagina -> {out} ({len(html)//1024} KB) · total {time.time()-t0:.0f}s", flush=True)


def _suggest(tick_data, S, SC, drop_req):
    """Candidatos de compra con un selector. S en % (entry, max_dev); SC en fracciones."""
    out = []
    for t, d in tick_data.items():
        if d.get("yhat") is None or d.get("r2") is None:
            continue
        if not (d["r2"] >= SC["min_r2"] and d["sigma"] <= SC["max_sigma"] and SC["min_growth"] <= d["growth"] <= SC["max_growth"] and d["adr"] <= SC["max_adr"]):
            continue
        dev = (d["close"] / d["yhat"] - 1) * 100
        drop = (d["close"] / d["prev"] - 1) * 100 if d.get("prev") else None
        if not (-S["max_dev"] < dev <= -S["entry"]):
            continue
        if drop is not None and drop <= -S["sl"]:
            continue   # cuchillo cayendo
        if drop_req and not (drop is not None and drop <= -drop_req):
            continue
        out.append((dev, t, drop, d["close"], d["yhat"]))
    return sorted(out)


def write_slack_summary(D, tick_data, positions, est, states, scr_raw):
    """Escribe slack_summary.md: estado de mi cartera, decisiones segun reglas y sugerencias por selector.
    Presets del usuario: lee user_presets.json si existe ({"1": {S, SC, drop}, ...}, como en la base de datos del artifact)."""
    import poslog
    from names import NAMES
    nm = lambda t: f"{t} ({NAMES.get(t, t)})"
    L = [f"*Prophet Trader · sesion {D}*", ""]
    cfg = dict(sl=est["sl_pct_pct"], exit_dev=est["exit_dev_pct"], hold=est["max_hold"], date=D)
    L.append("*Mi cartera*")
    open_p = [p for p in positions if not p.get("cerrada")]
    if not open_p:
        L.append("Sin posiciones abiertas.")
    tot_cap = tot_pnl = 0.0
    for p in open_p:
        d = tick_data.get(p["ticker"], {})
        c, y = d.get("close"), d.get("yhat")
        pnl = (c / p["precio_entrada"] - 1) * 100 if c else None
        v, code = poslog.verdict(p, c, y, cfg)
        tot_cap += p["capital"]; tot_pnl += p["capital"] * (pnl or 0) / 100
        L.append(f"• {nm(p['ticker'])} · {p.get('plataforma', '')} {p.get('cantidad') or ''} acc. · entrada {p['precio_entrada']:.2f} $ · hoy {c:.2f} $ · "
                 f"P&L {pnl:+.1f} % ({p['capital'] * (pnl or 0) / 100:+.2f} €)" if c else f"• {nm(p['ticker'])}: sin precio")
        L.append(f"   → {'*VENDER* ' if code in ('sl', 'tp', 'hold') else ''}{v}")
    if open_p:
        L.append(f"Total: {tot_cap:.2f} € invertidos · P&L latente {tot_pnl:+.2f} €")
    L.append("")
    selectors = []
    for key, c in paper.STRATEGIES.items():
        prof = paper.SCREEN_PROFILES[-1]
        selectors.append((c["name"], dict(entry=c["entry_dev"], max_dev=c["max_dev"], sl=c["sl"]),
                          dict(min_r2=prof["min_r2"], max_sigma=prof["max_sigma"], min_growth=prof["min_growth"], max_growth=prof["max_growth"], max_adr=prof["max_adr"]),
                          c["drop"]))
    up_path = os.path.join(APP_DIR, "user_presets.json")
    if os.path.exists(up_path):
        try:
            for n, p in sorted(json.load(open(up_path, encoding="utf-8")).items()):
                S, SC = p["S"], p["SC"]
                selectors.append((f"Mi preset {n}", dict(entry=S["entry"], max_dev=S["max_dev"], sl=S["sl"]),
                                  dict(min_r2=SC["min_r2"], max_sigma=SC["max_sigma"] / 100, min_growth=SC["min_growth"] / 100,
                                       max_growth=SC["max_growth"] / 100, max_adr=SC["max_adr"] / 100), p.get("drop")))
        except Exception as e:
            L.append(f"(no pude leer los presets guardados: {e})")
    L.append("*Sugerencias de compra para la apertura de mañana*")
    for name, S, SC, drop in selectors:
        c = _suggest(tick_data, S, SC, drop)
        L.append(f"_{name}_ (entrada {S['entry']:.0f}-{S['max_dev']:.0f} % bajo yhat{', caida dia ≥ ' + str(drop) + ' %' if drop else ''}): "
                 + (", ".join(f"{nm(t)} {dev:+.1f} %" + (f" (dia {dr:+.1f} %)" if dr is not None else "") + f" · {cl:.2f} $" for dev, t, dr, cl, yh in c[:6])
                    if c else "ninguna"))
    L.append("")
    L.append("*Carteras de papel de Claude*")
    for key, st in states.items():
        s = paper.summary(st)
        last = [x for x in (st["log"][-1]["events"] if st["log"] else []) if "orden" in x or "COMPRA" in x or "VENTA" in x]
        L.append(f"• {paper.STRATEGIES[key]['name']}: {s['equity']:.2f} $ ({s['ret']:+.1%}) · " + ("; ".join(last) if last else "sin operaciones hoy"))
    L.append("")
    L.append("Pagina: https://claude.ai/artifact/8Qs2ktpg4uhWNx6XmbVcnY · Simulacion y reglas mecanicas, no recomendacion de inversion.")
    txt = "\n".join(L)
    with open(os.path.join(APP_DIR, "slack_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[daily] resumen para Slack -> slack_summary.md ({len(txt)} caracteres)", flush=True)


if __name__ == "__main__":
    main()
