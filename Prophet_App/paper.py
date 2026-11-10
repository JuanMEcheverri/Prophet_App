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

Capital: 1.000 $ en una sola bolsa compartida (USD, sin cambio EUR/USD), hasta 3 posiciones a la vez,
ACCIONES ENTERAS. Tamaño de cada compra: objetivo = efectivo disponible / huecos libres, redondeado al
numero entero de acciones mas cercano (minimo 1), sin pasarse del efectivo. Ej.: 333 $ objetivo y la
accion a 200 $ -> 2 acciones (400 $); a 500 $ -> 1 accion; a 1.200 $ -> no se puede comprar.
1 $ por operacion, 5 pb de deslizamiento.
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

# Screening CONGELADO por fechas dentro del estado, para que un cambio posterior de umbrales no
# reescriba la elegibilidad del pasado (hallazgo de la auditoria del 2026-11-04): hasta el 4-nov se
# aplica el perfil que estaba en vigor antes de la ventana simulada; desde el 5-nov, el perfil
# revisado (elegido con datos hasta el 4-nov). Lo verdaderamente fuera de muestra empieza el 5-nov.
SCREEN_PROFILES = [
    {"from": "2026-09-01", "name": "perfil previo a la ventana (R2>=0.68, sigma<=6%, crecimiento 2-35%, ADR<=4%)",
     "min_r2": 0.68, "max_sigma": 0.06, "min_growth": 0.02, "max_growth": 0.35, "max_adr": 0.04},
    {"from": "2026-11-05", "name": "perfil revisado el 2026-11-04 (ADR<=4%, crecimiento<=20%, sin minimo; R2 y sigma desactivados)",
     "min_r2": 0.50, "max_sigma": 0.15, "min_growth": -0.50, "max_growth": 0.20, "max_adr": 0.04},
]
OOS_FROM = "2026-11-05"


def new_state(strategy):
    return {
        "strategy": strategy, "last_date": None, "cash": CAPITAL,
        "screening": [dict(p) for p in SCREEN_PROFILES], "oos_from": OOS_FROM,
        "slots": [{"position": None, "pending": None} for _ in range(SLOTS)],
        "closed": [], "equity": [], "log": [],
    }


def _migrate(state):
    """Estados antiguos (efectivo por slot, acciones fraccionadas, sin screening congelado)."""
    if "cash" not in state:
        state["cash"] = sum(s.pop("cash", 0.0) for s in state["slots"])
    for s in state["slots"]:
        s.pop("cash", None)
    state.setdefault("screening", [dict(p) for p in SCREEN_PROFILES])
    state.setdefault("oos_from", OOS_FROM)
    return state


def screen_profile(state, date):
    """Perfil de screening vigente en `date` (el ultimo cuyo 'from' <= date)."""
    date_s = date if isinstance(date, str) else date.isoformat()
    prof = state["screening"][0]
    for p in state["screening"]:
        if p["from"] <= date_s:
            prof = p
    return prof


def passes_screen(m, prof):
    """m: dict con r2, sigma, growth, adr (fracciones). None -> no pasa."""
    if m is None or any(m.get(k) is None for k in ("r2", "sigma", "growth", "adr")):
        return False
    return (m["r2"] >= prof["min_r2"] and m["sigma"] <= prof["max_sigma"]
            and prof["min_growth"] <= m["growth"] <= prof["max_growth"] and m["adr"] <= prof["max_adr"])


def _d(s):
    return _date.fromisoformat(s) if isinstance(s, str) else s


def size_order(cash, price, free_slots):
    """Acciones enteras a comprar con `cash` disponible repartido entre `free_slots` huecos."""
    if free_slots <= 0 or price <= 0 or cash - FEE < price:
        return 0
    target = cash / free_slots
    shares = max(1, int(round(target / price)))
    while shares > 0 and shares * price + FEE > cash:
        shares -= 1
    return shares


def step(state, date, day_data):
    """Avanza un dia. day_data: {ticker: {open, close, prev_close, yhat, screen}} con datos solo de `date`.
    1) ejecuta a la apertura de `date` las ordenes decididas el dia anterior;
    2) marca la cartera al cierre; 3) decide salidas y entradas con el cierre de `date` (quedan pendientes)."""
    _migrate(state)
    cfg = STRATEGIES[state["strategy"]]
    date_s = date.isoformat() if not isinstance(date, str) else date
    date = _d(date_s)
    if state["last_date"] and _d(state["last_date"]) >= date:
        return state
    log = []
    # 1) ejecutar ordenes pendientes a la apertura de hoy (primero ventas, para liberar efectivo)
    order = sorted(range(len(state["slots"])), key=lambda i: 0 if (state["slots"][i]["pending"] or {}).get("side") == "sell" else 1)
    for i in order:
        slot = state["slots"][i]
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
            free = sum(1 for s in state["slots"] if s["position"] is None)
            shares = size_order(state["cash"], px, free)
            if shares == 0:
                slot["pending"] = None
                log.append(f"slot {i+1}: orden de compra de {o['ticker']} cancelada: a {px:.2f} $ no alcanza el efectivo ({state['cash']:.2f} $) ni para 1 accion")
                continue
            cost = shares * px + FEE
            state["cash"] -= cost
            slot["position"] = dict(ticker=o["ticker"], signal_date=o["signal_date"], signal_close=o["signal_close"],
                                    signal_dev=o["dev"], signal_drop=o.get("drop"), fill_date=date_s,
                                    fill_px=round(px, 4), shares=shares, cash_in=round(cost, 2))
            log.append(f"slot {i+1}: COMPRA {shares} x {o['ticker']} a {px:.2f} = {cost:.2f} $ (señal {o['signal_date']}, {o['dev']:+.1f}% bajo yhat); efectivo {state['cash']:.2f} $")
        else:
            p = slot["position"]
            px = px_open * (1 - SLIP)
            proceeds = p["shares"] * px - FEE
            pnl = proceeds - p["cash_in"]
            state["closed"].append({**p, "exit_date": date_s, "exit_px": round(px, 4), "exit_reason": o["reason"],
                                    "days": (date - _d(p["fill_date"])).days, "pnl_pct": round(px / p["fill_px"] - 1, 5),
                                    "pnl_cash": round(pnl, 2), "cash_out": round(proceeds, 2)})
            state["cash"] += proceeds
            slot["position"] = None
            log.append(f"slot {i+1}: VENTA {p['shares']} x {p['ticker']} a {px:.2f} por {o['reason']} ({px / p['fill_px'] - 1:+.1%}, {pnl:+.2f} $, {(date - _d(p['fill_date'])).days} d); efectivo {state['cash']:.2f} $")
        slot["pending"] = None
    # 2) marcar al cierre de hoy
    eq = state["cash"]
    for slot in state["slots"]:
        if slot["position"]:
            dd = day_data.get(slot["position"]["ticker"])
            px = dd["close"] if dd and dd.get("close") is not None else slot["position"]["fill_px"]
            eq += slot["position"]["shares"] * px
    state["equity"].append({"date": date_s, "equity": round(eq, 2), "oos": date_s >= state["oos_from"]})
    # 3) decisiones con el cierre de hoy -> ordenes para la apertura de mañana
    prof = screen_profile(state, date_s)
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
            if t in held or dd.get("close") is None or not dd.get("yhat"):
                continue
            # elegibilidad con el perfil congelado para esta fecha (si vienen metricas); si no, la bandera externa
            eligible = passes_screen(dd.get("metrics"), prof) if dd.get("metrics") is not None else bool(dd.get("screen"))
            if not eligible:
                continue
            dev = (dd["close"] / dd["yhat"] - 1) * 100
            if not (-cfg["max_dev"] < dev <= -cfg["entry_dev"]):
                continue
            drop = None
            if dd.get("prev_close"):
                drop = (dd["close"] / dd["prev_close"] - 1) * 100
            if cfg["drop"] is not None and (drop is None or drop > -cfg["drop"]):
                continue
            if dd["close"] > state["cash"] - FEE:
                continue   # ni una accion entera con el efectivo actual
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
            log.append(f"slot {i+1}: orden de COMPRA {t} ({dev:+.1f}% bajo yhat" + (f", caida del dia {drop:+.1f}%" if drop is not None else "") + f"; cierre {close:.2f} $) para la apertura siguiente")
        if not cands:
            log.append("sin candidatos que cumplan el criterio" + (" (o sin efectivo para una accion entera)" if state["cash"] < 300 else ""))
    # estado de la cartera al cierre, siempre (para que ningun dia del diario quede vacio)
    held_txt = []
    for slot in state["slots"]:
        p = slot["position"]
        if not p:
            continue
        dd = day_data.get(p["ticker"]) or {}
        c = dd.get("close")
        d = (date - _d(p["fill_date"])).days
        held_txt.append(f"{p['ticker']} {p['shares']} acc. {p['fill_px']:.2f}->{c:.2f} $ ({c / p['fill_px'] - 1:+.1%}, {d} d)" if c
                        else f"{p['ticker']} {p['shares']} acc. (sin cierre)")
    if not any("orden de" in x or "COMPRA" in x or "VENTA" in x for x in log):
        log.append("sin operaciones: " + ("mantener todas las posiciones (ninguna toca hold, stop loss ni objetivo)" if held_txt else "sin posiciones")
                   + ("" if free else "; sin huecos libres para comprar"))
    log.append(f"estado al cierre: valor {eq:.2f} $, efectivo {state['cash']:.2f} $" + (" · " + "; ".join(held_txt) if held_txt else ""))
    state["log"].append({"date": date_s, "events": log, "oos": date_s >= state["oos_from"]})
    state["log"] = state["log"][-400:]
    state["last_date"] = date_s
    return state


def summary(state):
    _migrate(state)
    closed = state["closed"]
    n = len(closed)
    wins = sum(1 for c in closed if c["pnl_cash"] > 0)
    eq = state["equity"][-1]["equity"] if state["equity"] else CAPITAL
    return dict(equity=eq, ret=eq / CAPITAL - 1, n_closed=n, win_rate=(wins / n if n else None),
                pnl_closed=sum(c["pnl_cash"] for c in closed), cash=state["cash"],
                fees=FEE * (2 * n + sum(1 for s in state["slots"] if s["position"])))
