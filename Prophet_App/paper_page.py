"""Genera paper_page.html: pagina publica (artifact) con las dos carteras de papel y las señales del dia."""
import html, json
from datetime import date


def _fmt_pct(x, d=1):
    return "—" if x is None else f"{x*100:+.{d}f} %".replace(".", ",")


def _fmt_usd(x):
    return "—" if x is None else f"{x:,.2f} $".replace(",", "X").replace(".", ",").replace("X", ".")


def _sparkline(points, w=320, h=70, base=1000.0):
    if len(points) < 2:
        return ""
    ys = [p["equity"] for p in points]
    lo, hi = min(ys + [base]), max(ys + [base])
    if hi == lo:
        hi = lo + 1
    def X(i): return 4 + i * (w - 8) / (len(ys) - 1)
    def Y(v): return 4 + (hi - v) * (h - 8) / (hi - lo)
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
    area = path + f" L{X(len(ys)-1):.1f},{h-4} L4,{h-4} Z"
    color = "var(--ok)" if ys[-1] >= base else "var(--bad)"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="Evolucion de la cartera">'
            f'<path d="{area}" fill="{color}" opacity="0.12"/>'
            f'<line x1="4" x2="{w-4}" y1="{Y(base):.1f}" y2="{Y(base):.1f}" stroke="var(--muted)" stroke-dasharray="3 3" stroke-width="1"/>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="{X(len(ys)-1):.1f}" cy="{Y(ys[-1]):.1f}" r="3" fill="{color}"/></svg>')


def _table(headers, rows, cls=""):
    if not rows:
        return '<p class="muted">Nada que mostrar todavia.</p>'
    th = "".join(f"<th{' class=\"num\"' if h[1] else ''}>{html.escape(h[0])}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = []
        for (hname, num), v in zip(headers, r):
            cls_td = []
            if num: cls_td.append("num")
            if isinstance(v, tuple):   # (texto, clase)
                v, extra = v; cls_td.append(extra)
            tds.append(f"<td class=\"{' '.join(cls_td)}\">{html.escape(str(v))}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def _pnl_cls(x):
    return "" if x is None else ("good" if x > 0 else "poor" if x < 0 else "")


def build(states, day_data, D, prices_now, cfgs, version, screening):
    """states: {key: state}; day_data: {ticker: {...}} de la sesion D; prices_now: {ticker: close}."""
    import paper
    sections = []
    for key, state in states.items():
        cfg = cfgs[key]
        s = paper.summary(state)
        rows = []
        for i, slot in enumerate(state["slots"]):
            p, o = slot["position"], slot["pending"]
            if p:
                cpx = prices_now.get(p["ticker"])
                pnl = (cpx / p["fill_px"] - 1) if cpx else None
                yh = day_data.get(p["ticker"], {}).get("yhat")
                rows.append([i + 1, "abierta" + (f" · venta pendiente ({o['reason']})" if o and o["side"] == "sell" else ""),
                             p["ticker"], p["signal_date"], f"{p['signal_dev']:+.1f} %",
                             (f"{p['signal_drop']:+.1f} %" if p.get("signal_drop") is not None else "—"),
                             p["fill_date"], _fmt_usd(p["fill_px"]), _fmt_usd(cpx), (_fmt_pct(pnl), _pnl_cls(pnl)),
                             (date.fromisoformat(D) - date.fromisoformat(p["fill_date"])).days,
                             _fmt_usd(yh * (1 + cfg["exit_dev"] / 100) if yh else None), _fmt_usd(p["fill_px"] * (1 - cfg["sl"] / 100))])
            elif o:
                rows.append([i + 1, "compra pendiente (apertura siguiente)", o["ticker"], o["signal_date"], f"{o['dev']:+.1f} %",
                             (f"{o['drop']:+.1f} %" if o.get("drop") is not None else "—"), "—", "—", _fmt_usd(prices_now.get(o["ticker"])), "—", "—", "—", "—"])
            else:
                rows.append([i + 1, f"libre · {_fmt_usd(slot['cash'])}", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"])
        open_tbl = _table([("Slot", 1), ("Estado", 0), ("Ticker", 0), ("Señal", 0), ("Bajo yhat", 1), ("Caida dia", 1), ("Compra", 0),
                           ("Precio compra", 1), ("Precio hoy", 1), ("P&L", 1), ("Dias", 1), ("Vender si ≥", 1), ("Stop loss", 1)], rows)
        closed_rows = []
        for c in sorted(state["closed"], key=lambda c: c["exit_date"], reverse=True):
            closed_rows.append([c["ticker"], c["signal_date"], f"{c['signal_dev']:+.1f} %", c["fill_date"], _fmt_usd(c["fill_px"]),
                                c["exit_date"], _fmt_usd(c["exit_px"]), c["exit_reason"], c["days"],
                                (_fmt_pct(c["pnl_pct"]), _pnl_cls(c["pnl_pct"])), (_fmt_usd(c["pnl_cash"]), _pnl_cls(c["pnl_cash"]))])
        closed_tbl = _table([("Ticker", 0), ("Señal", 0), ("Bajo yhat", 1), ("Compra", 0), ("Precio", 1), ("Venta", 0), ("Precio", 1),
                             ("Motivo", 0), ("Dias", 1), ("P&L", 1), ("P&L $", 1)], closed_rows)
        log_html = "".join(
            f"<div class=\"day\"><b>{html.escape(e['date'])}</b><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in e["events"]) + "</ul></div>"
            for e in reversed(state["log"][-8:]))
        entry_txt = (f"precio entre {cfg['entry_dev']:.0f} % y {cfg['max_dev']:.0f} % por debajo de yhat"
                     + (f" <b>y</b> caida del dia ≥ {cfg['drop']:.0f} %" if cfg["drop"] else ""))
        sections.append(f"""
  <section class="strat">
    <h2>{html.escape(cfg['name'])}</h2>
    <p class="rule">Entrada: {entry_txt}. Salida (en este orden): {cfg['hold']} dias → stop loss {cfg['sl']:.0f} % → precio ≥ yhat +{cfg['exit_dev']:.0f} %. {paper.CAPITAL:.0f} $ en {paper.SLOTS} slots, {paper.FEE:.0f} $ por operacion.</p>
    <div class="kpis">
      <div class="kpi"><div class="l">Valor cartera</div><div class="v {_pnl_cls(s['ret'])}">{_fmt_usd(s['equity'])}</div><div class="n">{_fmt_pct(s['ret'])} desde el inicio</div></div>
      <div class="kpi"><div class="l">Operaciones cerradas</div><div class="v">{s['n_closed']}</div><div class="n">acierto {('—' if s['win_rate'] is None else f"{s['win_rate']*100:.0f} %")}</div></div>
      <div class="kpi"><div class="l">P&amp;L cerrado</div><div class="v {_pnl_cls(s['pnl_closed'])}">{_fmt_usd(s['pnl_closed'])}</div><div class="n">comisiones pagadas {_fmt_usd(s['fees'])}</div></div>
      <div class="kpi spark"><div class="l">Evolucion</div>{_sparkline(state['equity'])}</div>
    </div>
    <h3>Posiciones y ordenes</h3>{open_tbl}
    <h3>Operaciones cerradas</h3>{closed_tbl}
    <details><summary>Diario de decisiones (ultimos dias)</summary>{log_html}</details>
  </section>""")

    # señales del dia: candidatos por debajo de yhat, ordenados por desviacion
    cands = []
    for t, dd in day_data.items():
        if dd.get("yhat") and dd.get("close"):
            dev = (dd["close"] / dd["yhat"] - 1) * 100
            drop = ((dd["close"] / dd["prev_close"] - 1) * 100) if dd.get("prev_close") else None
            cands.append((dev, t, drop, dd["close"], dd["yhat"], dd.get("screen")))
    cands.sort()
    sig_rows = []
    for dev, t, drop, close, yh, scr in cands[:25]:
        tag = "🟢 compra" if (scr and -15 < dev <= -4) else ("⚠️ muy bajo" if dev <= -15 else "")
        sig_rows.append([t, _fmt_usd(close), _fmt_usd(yh), (f"{dev:+.1f} %", _pnl_cls(dev)), (f"{drop:+.1f} %" if drop is not None else "—"),
                         "sí" if scr else "no", tag])
    sig_tbl = _table([("Ticker", 0), ("Cierre", 1), ("yhat", 1), ("Bajo/sobre yhat", 1), ("Caida del dia", 1), ("Pasa screening", 0), ("Señal", 0)], sig_rows)
    n_pass = sum(1 for dd in day_data.values() if dd.get("screen"))

    return f"""<title>Prophet Trader · Cartera de papel</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{ --bg:#f3f4f1; --panel:#fff; --fg:#1d2229; --muted:#5d6673; --line:#d9dcd6; --accent:#b86a00; --ok:#2c7a4b; --bad:#b23a3a;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#15181c; --panel:#1d2127; --fg:#e6e8e3; --muted:#9aa3ad; --line:#2c3139; --accent:#f0a23a; --ok:#6fcf93; --bad:#ef8a8a; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#15181c; --panel:#1d2127; --fg:#e6e8e3; --muted:#9aa3ad; --line:#2c3139; --accent:#f0a23a; --ok:#6fcf93; --bad:#ef8a8a; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.5; padding-block:24px 48px;
  padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); }}
.wrap {{ max-width:960px; margin:0 auto; }}
h1 {{ font-family:var(--mono); font-size:22px; font-weight:600; margin:0 0 4px; }}
h2 {{ font-size:18px; margin:32px 0 4px; }} h3 {{ font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:20px 0 8px; }}
.sub, .muted, .rule, .n {{ color:var(--muted); }} .sub {{ margin:0 0 18px; font-size:14px; }} .rule {{ font-size:14px; margin:0 0 12px; max-width:75ch; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); padding:12px 14px; min-width:0; }}
.kpi .l {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
.kpi .v {{ font-family:var(--mono); font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }} .kpi .n {{ font-size:12px; }}
.kpi.spark {{ grid-column: span 2; }}
.tbl {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }} th, td {{ padding:7px 10px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }} td.num, th.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
tr:last-child td {{ border-bottom:0; }} .good {{ color:var(--ok); font-weight:500; }} .poor {{ color:var(--bad); font-weight:500; }}
details {{ margin-top:12px; }} summary {{ cursor:pointer; color:var(--muted); font-size:14px; }} .day {{ font-size:13px; margin:8px 0; }} .day ul {{ margin:2px 0 0 18px; padding:0; }}
.note {{ font-size:13px; color:var(--muted); max-width:70ch; }} .foot {{ margin-top:28px; font-family:var(--mono); font-size:12px; color:var(--muted); }}
.strat {{ background:transparent; }}
</style>
<div class="wrap">
  <h1>Prophet Trader · Cartera de papel</h1>
  <p class="sub">Ultima sesion procesada: <b>{html.escape(D)}</b> · app v{html.escape(str(version))} · se actualiza cada dia laborable tras el cierre de EE.UU. Cada decision usa solo los datos de ese dia y se ejecuta a la apertura siguiente.</p>
  {"".join(sections)}
  <h2>Señales de la sesion {html.escape(D)}</h2>
  <p class="rule">Las 25 acciones mas por debajo de su precio justo (yhat). Pasan el screening {n_pass} de {len(day_data)} (ADR ≤ {screening['max_adr_pct']} %, crecimiento {screening['min_growth_pct']}..{screening['max_growth_pct']} %/año).</p>
  {sig_tbl}
  <p class="note">Esto es una simulacion con 1.000 $ ficticios para comprobar en tiempo real, sin trampas, si las reglas funcionan. No es una recomendacion de compra. En el backtest de 5 años ninguna de las dos estrategias batio a QQQ de forma clara.</p>
  <div class="foot">Fuente: Yahoo Finance (cierres ajustados), Prophet con ventana de 1 año. Repositorio: JuanMEcheverri/Prophet_App.</div>
</div>
"""
