"""Diario de decisiones de MIS posiciones (las reales, positions.json).

Una entrada por sesion de mercado, idempotente: para cada posicion abierta en esa fecha, precio de
cierre, P&L, dias y veredicto de las reglas de la app (hold, stop loss sobre la entrada, salida cuando
el precio supera yhat x (1 + salida)); mas las compras y ventas registradas ese dia.
"""
from datetime import date as _date

MAX_DAYS = 400


def _d(s):
    return _date.fromisoformat(s) if isinstance(s, str) else s


def verdict(p, close, yhat, cfg):
    """cfg: dict con sl (%), exit_dev (%), hold (dias). Devuelve (texto, codigo)."""
    days = (_d(cfg["date"]) - _d(p["fecha_entrada"])).days
    sl_px = p["precio_entrada"] * (1 - cfg["sl"] / 100)
    tp_px = yhat * (1 + cfg["exit_dev"] / 100) if yhat else None
    if days >= cfg["hold"]:
        return f"⏰ VENCIDA: {days} dias >= {cfg['hold']}. La regla dice vender a la apertura siguiente", "hold"
    if close is not None and close <= sl_px:
        return f"🛑 STOP LOSS: cierre {close:.2f} $ <= {sl_px:.2f} $ (−{cfg['sl']:.0f} % de la entrada). La regla dice vender", "sl"
    if tp_px and close is not None and close >= tp_px:
        return f"✅ OBJETIVO: cierre {close:.2f} $ >= yhat +{cfg['exit_dev']:.0f} % ({tp_px:.2f} $). La regla dice vender", "tp"
    extra = []
    if close is not None:
        extra.append(f"SL a {((sl_px / close) - 1) * 100:+.1f} %")
        if tp_px:
            extra.append(f"objetivo a {((tp_px / close) - 1) * 100:+.1f} %")
    extra.append(f"quedan {cfg['hold'] - days} dias")
    return "⏳ mantener (" + ", ".join(extra) + ")", "keep"


def update_log(log, date_s, positions, day_data, cfg):
    """log: lista de {date, events}. day_data: {ticker: {close, yhat}}. Añade la sesion si no existe."""
    if any(e["date"] == date_s for e in log):
        return log, False
    cfg = {**cfg, "date": date_s}
    events = []
    for p in positions:
        t = p["ticker"]
        if p["fecha_entrada"] == date_s:
            events.append(f"COMPRA registrada: {t} a {p['precio_entrada']:.2f} $"
                          + (f" ({p['precio_entrada_eur']:.2f} €)" if p.get("precio_entrada_eur") else "")
                          + f", capital {p['capital']:.2f} €"
                          + (f", {p['cantidad']:.4f} acciones" if p.get("cantidad") else "")
                          + (f" en {p['plataforma']}" if p.get("plataforma") else "")
                          + (f" · {p['nota']}" if p.get("nota") else ""))
        if p.get("cerrada") and p.get("fecha_salida") == date_s:
            events.append(f"VENTA registrada: {t} a {p['precio_salida']:.2f} $ → P&L {p.get('pnl_eur', 0):+.2f} € ({p.get('pnl_pct', 0):+.1f} %)"
                          + (f" · {p['motivo_salida']}" if p.get("motivo_salida") else ""))
        is_open = (not p.get("cerrada")) or (p.get("fecha_salida") and p["fecha_salida"] > date_s)
        if is_open and p["fecha_entrada"] <= date_s:
            dd = day_data.get(t, {})
            close, yhat = dd.get("close"), dd.get("yhat")
            pnl = (close / p["precio_entrada"] - 1) * 100 if close else None
            v, _ = verdict(p, close, yhat, cfg)
            events.append(f"{t}: cierre {close:.2f} $" if close else f"{t}: sin cierre"
                          + "")
            events[-1] += (f", P&L {pnl:+.1f} % ({p['capital'] * pnl / 100:+.2f} €)" if pnl is not None else "") \
                + (f", yhat {yhat:.2f} $ ({(close / yhat - 1) * 100:+.1f} %)" if (yhat and close) else "") + " → " + v
    if not events:
        events.append("sin posiciones abiertas")
    log.append({"date": date_s, "events": events})
    log.sort(key=lambda e: e["date"])
    return log[-MAX_DAYS:], True
