"""Nivel 2: motor de slots con capital limitado (el que vive el usuario).

Cada mañana del dia X (dia de accion):
  1) salidas, en el orden de la app sobre el cierre de X-1: max_hold, SL, TP (o devx); venta a open(X)
  2) entradas en slots libres: candidatos con dev en (-max_dev, -entry], screening, no en cartera,
     no vendidos esa misma mañana (corrige la recompra el mismo dia de los scripts 13/14);
     los mas negativos primero; compra a open(X)
Señal usada en X: post_close -> dev de close(X-1) con yhat vigente en X-1; pre_open -> dev calculado
la mañana de X (close(X-1) vs yhat pronosticado para X). Contabilidad en EUR con EURUSD, fee y slippage.
Capital se reinicia a 1.000 EUR al inicio de cada periodo; ademas una curva continua para drawdown.
"""
import os, json, time
import numpy as np
import pandas as pd

from . import common as C
from .level1 import KEY_CONFIGS, _ffill

INITIAL = 1000.0


def _bench_returns(wide, cal, start, end, tickers):
    cl = wide["close"].reindex(cal)
    w = cl[(cl.index >= start) & (cl.index <= end)]
    if len(w) < 2:
        return {}
    out = {"qqq": float(w["QQQ"].iloc[-1] / w["QQQ"].iloc[0] - 1)}
    ew = (w[tickers].ffill().iloc[-1] / w[tickers].bfill().iloc[0] - 1)
    out["ew_universe"] = float(ew.mean())
    l7 = [t for t in C.LIST_07 if t in w.columns]
    out["ew_list07"] = float((w[l7].ffill().iloc[-1] / w[l7].bfill().iloc[0] - 1).mean())
    # mismos benchmarks en EUR (un inversor en EUR que compra QQQ sufre el mismo EUR/USD que la estrategia)
    fxw = w["EURUSD=X"].ffill().bfill()
    fx_ratio = float(fxw.iloc[0] / fxw.iloc[-1])
    out["qqq_eur"] = (1 + out["qqq"]) * fx_ratio - 1
    out["ew_universe_eur"] = (1 + out["ew_universe"]) * fx_ratio - 1
    return out


def simulate(cfg, proto, screen_on, universe_fn, sig_tbl, wide, cal, fx, start, end, slots=3,
             seed=None, random_entries=False):
    """Un periodo [start, end]. universe_fn(date) -> set de tickers permitidos. Devuelve resumen + curva + tx."""
    days = cal[(cal >= start) & (cal <= end)]
    if len(days) < 5:
        return None
    pos_of = {d: i for i, d in enumerate(cal)}
    op = wide["open"].reindex(cal); cl = wide["close"].reindex(cal)
    openA = {t: _ffill(op[t].values.astype(float)) for t in C.TICKERS if t in op.columns}
    closeA = {t: _ffill(cl[t].values.astype(float)) for t in C.TICKERS if t in cl.columns}
    open_raw = {t: op[t].values.astype(float) for t in C.TICKERS if t in op.columns}
    qqq_close = _ffill(cl["QQQ"].values.astype(float))
    # dev por dia de accion X: post_close usa fila X-1 (dev_post); pre_open usa fila X (dev_pre)
    scol = {True: "screen", "live": "screen_live"}.get(screen_on, "screen")
    if proto == "post_close":
        s = sig_tbl[["ticker", "pos", "dev_post", scol]].copy()
        s["x"] = s["pos"] + 1
        s = s.rename(columns={"dev_post": "dev", scol: "screen"})
    else:
        s = sig_tbl[["ticker", "pos", "dev_pre", scol + "_pre"]].copy()
        s["x"] = s["pos"]
        s = s.rename(columns={"dev_pre": "dev", scol + "_pre": "screen"})
    s = s.dropna(subset=["dev"])
    if screen_on:
        s = s[s["screen"]]
    hold_lag = 1 if proto == "post_close" else 0   # usuario de tarde ve days>=hold la tarde anterior y vende a esta apertura
    s = s[(s["dev"] <= -cfg["entry"]) & (s["dev"] > -cfg["max_dev"])]
    dev_post_by = {t: g.set_index("pos")["dev_post"] for t, g in sig_tbl.groupby("ticker")} if cfg.get("devx") is not None else {}
    cands_by_x = {x: g.sort_values("dev") for x, g in s.groupby("x")}
    if random_entries:
        # mismo numero de señales por ticker en el periodo, en dias aleatorios
        rng = np.random.default_rng(seed)
        xs_period = [pos_of[d] for d in days]
        rows = []
        for t, g in s[(s["x"] >= pos_of[days[0]]) & (s["x"] <= pos_of[days[-1]])].groupby("ticker"):
            pick = rng.choice(xs_period, size=min(len(g), len(xs_period)), replace=False)
            rows.append(pd.DataFrame({"ticker": t, "x": pick, "dev": rng.permutation(g["dev"].values)[: len(pick)]}))
        s = pd.concat(rows, ignore_index=True) if rows else s.iloc[0:0]
        cands_by_x = {x: g.sort_values("dev") for x, g in s.groupby("x")}

    slot_cash = [INITIAL / slots] * slots
    holdings = [None] * slots       # dict(ticker, entry_px, entry_date, shares, entry_x)
    tx, equity, invested = [], [], []
    fee_total = 0.0
    for X in days:
        x = pos_of[X]
        fxX = fx[x]
        sold_today = set()
        # 1) salidas
        for i, h in enumerate(holdings):
            if h is None:
                continue
            t = h["ticker"]
            prev_close = closeA[t][x - 1]
            days_held = (X - h["entry_date"]).days
            reason = None
            if cfg["hold"] is not None and days_held - hold_lag >= cfg["hold"]:
                reason = "HOLD"
            elif cfg["sl"] is not None and prev_close <= h["entry_px"] * (1 - cfg["sl"] / 100):
                reason = "SL"
            elif cfg["tp"] is not None and prev_close >= h["entry_px"] * (1 + cfg["tp"] / 100):
                reason = "TP"
            elif cfg.get("devx") is not None:
                dv = dev_post_by[t].get(x - 1, np.nan)
                if dv >= cfg["devx"]:
                    reason = "DEVX"
            if reason:
                px = openA[t][x] * (1 - C.SLIP_BP / 1e4)
                usd = h["shares"] * px
                eur = usd / fxX * (1 - C.FX_SPREAD_BP / 1e4) - C.FEE_EUR
                fee_total += C.FEE_EUR
                tx.append(dict(slot=i, ticker=t, side="SELL", date=X, px=px, eur=eur, reason=reason,
                               days=days_held, pnl_pct=px / h["entry_px"] - 1, eur_pnl=eur - h["eur_in"]))
                slot_cash[i] = eur
                holdings[i] = None
                sold_today.add(t)
        # 2) entradas
        free = [i for i, h in enumerate(holdings) if h is None]
        if free and x in cands_by_x:
            held = {h["ticker"] for h in holdings if h}
            allowed = universe_fn(X)
            for _, r in cands_by_x[x].iterrows():
                if not free:
                    break
                t = r["ticker"]
                if t in held or t in sold_today or t not in allowed or np.isnan(open_raw[t][x]):
                    continue
                i = free.pop(0)
                px = openA[t][x] * (1 + C.SLIP_BP / 1e4)
                eur_in = slot_cash[i]
                usd = (eur_in - C.FEE_EUR) * fxX * (1 - C.FX_SPREAD_BP / 1e4)
                fee_total += C.FEE_EUR
                holdings[i] = dict(ticker=t, entry_px=px, entry_date=X, shares=usd / px, eur_in=eur_in)
                held.add(t)
                slot_cash[i] = 0.0
                tx.append(dict(slot=i, ticker=t, side="BUY", date=X, px=px, eur=eur_in, reason="",
                               days=0, pnl_pct=np.nan, eur_pnl=np.nan, dev=r["dev"]))
        # marcado diario
        inv = 0.0; eq = 0.0
        for i, h in enumerate(holdings):
            if h is None:
                eq += slot_cash[i]
            else:
                v = h["shares"] * closeA[h["ticker"]][x] / fxX
                eq += v; inv += v
        equity.append(eq); invested.append(inv / eq if eq > 0 else 0.0)
    # liquidar al ultimo cierre
    xl = pos_of[days[-1]]
    final = 0.0
    for i, h in enumerate(holdings):
        if h is None:
            final += slot_cash[i]
        else:
            px = closeA[h["ticker"]][xl]
            proceeds = h["shares"] * px / fx[xl] * (1 - C.FX_SPREAD_BP / 1e4) - C.FEE_EUR
            final += proceeds
            tx.append(dict(slot=i, ticker=h["ticker"], side="SELL", date=days[-1], px=px, eur=proceeds, reason="END",
                           days=(days[-1] - h["entry_date"]).days, pnl_pct=px / h["entry_px"] - 1, eur_pnl=proceeds - h["eur_in"]))
    eqs = np.array(equity)
    dd = float((eqs / np.maximum.accumulate(eqs) - 1).min()) if len(eqs) else 0.0
    inv_arr = np.array(invested)
    idx = [pos_of[d] for d in days]
    qr = qqq_close[idx] / fx[idx]            # QQQ en EUR, como la cartera
    qqq_daily = np.diff(qr) / qr[:-1]
    exp_adj = float(np.prod(1 + inv_arr[:-1] * qqq_daily) - 1)
    sells = [t for t in tx if t["side"] == "SELL" and t["reason"] != "END"]
    return dict(
        ret=final / INITIAL - 1, max_dd=dd, time_in_market=float(inv_arr.mean()),
        n_buys=sum(1 for t in tx if t["side"] == "BUY"), n_sells=len(sells),
        win_rate=float(np.mean([t["pnl_pct"] > 0 for t in sells])) if sells else np.nan,
        fees=fee_total, exp_adj_qqq=exp_adj,
        sh_hold=float(np.mean([t["reason"] == "HOLD" for t in sells])) if sells else np.nan,
        sh_sl=float(np.mean([t["reason"] == "SL" for t in sells])) if sells else np.nan,
        sh_tp=float(np.mean([t["reason"] == "TP" for t in sells])) if sells else np.nan,
    ), pd.Series(eqs, index=days), pd.DataFrame(tx)


def run(refit_every=5, n_random=20):
    t0 = time.time()
    wide = C.load_prices()
    cal = C.trading_calendar(wide)
    fx = wide["close"]["EURUSD=X"].reindex(cal).ffill().bfill().values.astype(float)
    sig = pd.read_parquet(os.path.join(C.RES_DIR, f"signals_refit{refit_every}.parquet"))
    last = cal[-1]
    cfgs = dict(KEY_CONFIGS)
    sl_path = os.path.join(C.RES_DIR, f"shortlist_refit{refit_every}.json")
    if os.path.exists(sl_path):
        for k, v in json.load(open(sl_path)).items():
            cfgs[k] = {f: (None if v[f] in (None, "None") else v[f]) for f in ["entry", "max_dev", "sl", "tp", "hold", "devx"]}
    universes = {"all87": lambda d: set(C.TICKERS), "list07": lambda d: set(C.LIST_07)}
    rows, curves, txs = [], [], []
    runs = [(name, cfg, proto, True, uni) for name, cfg in cfgs.items()
            for proto in ["post_close", "pre_open"] for uni in ["all87", "list07"]]
    runs += [("C0", cfgs["C0"], "post_close", False, "all87"),
             ("C_LIVE", cfgs["C_LIVE"], "post_close", "live", "all87")]
    for name, cfg, proto, scr, uni in runs:
        for slots in ([3, 5] if name == "C0" else [3]):
            for pname, a, b in C.PERIODS:
                start, end = pd.Timestamp(a), min(pd.Timestamp(b), last)
                res = simulate(cfg, proto, scr, universes[uni], sig, wide, cal, fx, start, end, slots=slots)
                if res is None:
                    continue
                summ, eq, tx = res
                bench = _bench_returns(wide, cal, start, end, C.TICKERS)
                row = dict(cfg=name, protocol=proto, screen=str(scr), universe=uni, slots=slots, period=pname, **summ, **bench)
                if name in ("C0", "HOLD15") and uni == "all87" and slots == 3 and scr is True:
                    rr, rt, re_ = [], [], []
                    for sdx in range(n_random):
                        r2 = simulate(cfg, proto, scr, universes[uni], sig, wide, cal, fx, start, end, slots=slots,
                                      seed=sdx, random_entries=True)
                        if r2:
                            rr.append(r2[0]["ret"]); rt.append(r2[0]["time_in_market"]); re_.append(r2[0]["exp_adj_qqq"])
                    row["random_mean"] = float(np.mean(rr)) if rr else np.nan
                    row["random_p95"] = float(np.percentile(rr, 95)) if rr else np.nan
                    row["random_tim"] = float(np.mean(rt)) if rt else np.nan
                    row["random_exp_adj"] = float(np.mean(re_)) if re_ else np.nan
                rows.append(row)
                eq = eq.rename("equity").reset_index().rename(columns={"index": "date"})
                eq["cfg"] = name; eq["protocol"] = proto; eq["universe"] = uni; eq["slots"] = slots; eq["period"] = pname; eq["screen"] = str(scr)
                curves.append(eq)
                if len(tx):
                    tx["cfg"] = name; tx["protocol"] = proto; tx["universe"] = uni; tx["slots"] = slots; tx["period"] = pname; tx["screen"] = str(scr)
                    txs.append(tx)
        print(f"  {name:10s} {proto:10s} scr={scr} {uni}: {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(C.RES_DIR, f"portfolio_summary_refit{refit_every}.parquet"), index=False)
    pd.concat(curves, ignore_index=True).to_parquet(os.path.join(C.RES_DIR, f"portfolio_curves_refit{refit_every}.parquet"), index=False)
    pd.concat(txs, ignore_index=True).to_parquet(os.path.join(C.RES_DIR, f"portfolio_tx_refit{refit_every}.parquet"), index=False)
    print(f"[portfolio] {len(out)} filas en {(time.time()-t0)/60:.1f} min", flush=True)
    return out
