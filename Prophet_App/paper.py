"""Cartera de papel: cada dia, con los datos de ESE dia, decide que comprar y que vender.

Garantia anti-trampa por construccion: `step(state, date, day_data)` solo recibe los datos
del dia `date` (cierre, apertura, yhat de un ajuste entrenado hasta ese cierre, caida del dia,
screening). Las decisiones tomadas tras el cierre de D se ejecutan a la apertura de D+1, que
solo se conoce cuando llega D+1 (orden 'pendiente' hasta entonces). No hay acceso a precios
posteriores ni a promedios calculados con el futuro.

Dos estrategias, mismas salidas (las del backtest 15), distinta entrada:
  bt15  : precio entre 4% y 15% por debajo de yhat
  combo : lo mismo Y ADEMAS el cierre de hoy cae >= 5% respecto a ayer (caida de un solo dia)
Salidas (sobre el cierre de D, ejecutadas a la apertura de D+1, en el orden de la app):
  hold 60 dias naturales -> SL 7% bajo el precio de entrada -> precio >= yhat x 1.07
Capital 1.000 (USD, sin cambio EUR/USD), 3 slots iguales, 1 por operacion, 5 pb de deslizamiento.
"""
from datetime import date as _date

STRATEGIES = {
    "bt15":  dict(name="Backtest 15", entry_dev=4.0, max_dev=15.0, exit_dev=7.0, sl=7.0, hold=60, drop=None),
    "combo": dict(name="Combinado (caida 1 dia >= 5% + 4% bajo yhat)", entry_dev=4.0, max_dev=15.0,
                  exit_dev=7.0, sl=7.0, hold=60, drop=5.0),
}
CAPITAL = 1000.0
SLOTS = 3
FEE = 1.0
SLIP = 0.0005


def new_state(strategy):
    return {
        "strategy": strategy, "last_date": None,
        "slots": [{"cash": CAPITAL / SLOTS, "position": None, "pending": None} for _ in range(SLOTS)],
        "closed": [], "equity": [], "log": [],
    }


def _d(s):
    return _date.fromisoformat(s) if isinstance(s, str) else s


def step(state, date, day_data):
    """Avanza un dia. day_data: {ticker: {open, close, prev_close, yhat, screen}} con datos solo de `date`.
    1) ejecuta a la apertura de `date` las ordenes decididas el dia anterior;
    2) marca la cartera al cierre; 3) decide salidas y entradas con el cierre de `date` (quedan pendientes)."""
    cfg = STRATEGIES[state["strategy"]]
    date_s = date.isoformat() if not isinstance(date, str) else date
    date = _d(date_s)
    if state["last_date"] and _d(state["last_date"]) >= date:
        return state
    log = []
    # 1) ejecutar ordenes pendientes a la apertura de hoy
    for i, slot in enumerate(state["slots"]):
        o = slot["pending"]
        if not o:
            continue
        dd = day_data.get(o["ticker"])
        px_open = dd.get("open") if dd else None
        if px_open is None:
            if o["side"] == "buy":
                slot["pending"] = None
                log.append(f"slot {i+1}: orden de compra de {o['ticker']} cancelada (sin apertura hoy)")
            continue   # venta: se reintenta al dia siguiente
        if o["side"] == "buy":
            px = px_open * (1 + SLIP)
            cash = slot["cash"] - FEE
            slot["position"] = dict(ticker=o["ticker"], signal_date=o["signal_date"], signal_close=o["signal_close"],
                                    signal_dev=o["dev"], signal_drop=o.get("drop"), fill_date=date_s,
                                    fill_px=round(px, 4), shares=cash / px, cash_in=slot["cash"])
            slot["cash"] = 0.0
            log.append(f"slot {i+1}: COMPRA {o['ticker']} a {px:.2f} (señal {o['signal_date']}, {o['dev']:+.1f}% bajo yhat)")
        else:
            p = slot["position"]
            px = px_open * (1 - SLIP)
            proceeds = p["shares"] * px - FEE
            pnl = proceeds - p["cash_in"]
            state["closed"].append({**p, "exit_date": date_s, "exit_px": round(px, 4), "exit_reason": o["reason"],
                                    "days": (date - _d(p["fill_date"])).days, "pnl_pct": round(px / p["fill_px"] - 1, 5),
                                    "pnl_cash": round(pnl, 2), "cash_out": round(proceeds, 2)})
            slot["cash"] = proceeds
            slot["position"] = None
            log.append(f"slot {i+1}: VENTA {p['ticker']} a {px:.2f} por {o['reason']} ({px / p['fill_px'] - 1:+.1%}, {(date - _d(p['fill_date'])).days} d)")
        slot["pending"] = None
    # 2) marcar al cierre de hoy
    eq = 0.0
    for slot in state["slots"]:
        if slot["position"]:
            dd = day_data.get(slot["position"]["ticker"])
            px = dd["close"] if dd and dd.get("close") is not None else slot["position"]["fill_px"]
            eq += slot["position"]["shares"] * px
        else:
            eq += slot["cash"]
    state["equity"].append({"date": date_s, "equity": round(eq, 2)})
    # 3) decisiones con el cierre de hoy -> ordenes para la apertura de mañana
    held = set()
    for i, slot in enumerate(state["slots"]):
        p = slot["position"]
        if not p:
            continue
        held.add(p["ticker"])
        dd = day_data.get(p["ticker"])
        if not dd or dd.get("close") is None:
            continue
        close = dd["close"]; yhat = dd.get("yhat")
        days = (date - _d(p["fill_date"])).days
        reason = None
        if days >= cfg["hold"]:
            reason = f"hold {cfg['hold']}d"
        elif close <= p["fill_px"] * (1 - cfg["sl"] / 100):
            reason = f"stop loss {cfg['sl']:.0f}%"
        elif yhat and close >= yhat * (1 + cfg["exit_dev"] / 100):
            reason = f"yhat +{cfg['exit_dev']:.0f}%"
        if reason:
            slot["pending"] = dict(side="sell", ticker=p["ticker"], reason=reason, decided=date_s)
            log.append(f"slot {i+1}: orden de VENTA {p['ticker']} ({reason}) para la apertura siguiente")
    free = [i for i, s in enumerate(state["slots"]) if s["position"] is None and s["pending"] is None]
    if free:
        cands = []
        for t, dd in day_data.items():
            if t in held or not dd.get("screen") or dd.get("close") is None or not dd.get("yhat"):
                continue
            dev = (dd["close"] / dd["yhat"] - 1) * 100
            if not (-cfg["max_dev"] < dev <= -cfg["entry_dev"]):
                continue
            drop = None
            if dd.get("prev_close"):
                drop = (dd["close"] / dd["prev_close"] - 1) * 100
            if cfg["drop"] is not None and (drop is None or drop > -cfg["drop"]):
                continue
            cands.append((dev, t, drop, dd["close"]))
        cands.sort()
        pending_t = {s["pending"]["ticker"] for s in state["slots"] if s["pending"] and s["pending"]["side"] == "buy"}
        for dev, t, drop, close in cands:
            if not free:
                break
            if t in pending_t:
                continue
            i = free.pop(0)
            state["slots"][i]["pending"] = dict(side="buy", ticker=t, signal_date=date_s, signal_close=round(close, 4),
                                                dev=round(dev, 2), drop=(round(drop, 2) if drop is not None else None))
            pending_t.add(t)
            log.append(f"slot {i+1}: orden de COMPRA {t} ({dev:+.1f}% bajo yhat" + (f", caida del dia {drop:+.1f}%" if drop is not None else "") + ") para la apertura siguiente")
        if not cands:
            log.append("sin candidatos que cumplan el criterio")
    state["log"].append({"date": date_s, "events": log})
    state["log"] = state["log"][-120:]
    state["last_date"] = date_s
    return state


def summary(state):
    closed = state["closed"]
    n = len(closed)
    wins = sum(1 for c in closed if c["pnl_cash"] > 0)
    eq = state["equity"][-1]["equity"] if state["equity"] else CAPITAL
    return dict(equity=eq, ret=eq / CAPITAL - 1, n_closed=n, win_rate=(wins / n if n else None),
                pnl_closed=sum(c["pnl_cash"] for c in closed), fees=FEE * (2 * n + sum(1 for s in state["slots"] if s["position"])))
