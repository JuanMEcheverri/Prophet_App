"""Nivel 1: operaciones independientes (una posicion por ticker) sobre el banco de ajustes.

Protocolos de señal (ambos salen del mismo banco):
  post_close: el usuario entrena tras el cierre de D. dev = close(D)/yhat_F(D) - 1 con F = ultimo
              fit_date <= D (k=0 cuando D es fit_date: yhat ajustado con el cierre de D incluido,
              exactamente lo que muestra la app). Compra a la apertura de D+1.
  pre_open:   el usuario entrena por la mañana de D (España) con cierres hasta D-1.
              dev = close(D-1)/yhat_F(D) - 1 con F = ultimo fit_date <= D-1 (yhat de D es pronostico).
              Compra a la apertura de D.
Salidas (orden de la app cada mañana sobre el cierre anterior: max_hold, SL, TP), venta a la
apertura siguiente. Costes: fee, slippage, FX EUR/USD. Cada operacion se compara con el control
incondicional (misma regla de salida entrando CUALQUIER dia en ese ticker y periodo) y con QQQ.
Controles sin Prophet: yhat sustituido por regresion lineal de 1 año sobre log(precio) y por SMA100.
No se simulan dos usos de la app: pulsar 'Entrenar' a media sesion de EE.UU. (get_prices devuelve un
precio intradia y el ajuste incluye una vela parcial) ni en fin de semana. El usuario debe actuar sobre
cierres completos (tarde tras el cierre, o mañana siguiente).
"""
import os, time, itertools
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import common as C
from .fit_bank import load_bank

ENTRY = [2, 3, 4, 5, 7]
MAXDEV = [8, 10, 12, 15, 20, 999]
SL = [3, 5, 7, 10, 15, None]
TP = [5, 7, 10, 15, 20, None]
HOLD = [5, 10, 15, 20, 30]
DEV_EXIT = [-1, 0, 2, 7, 20]     # salida anclada a yhat: vender cuando dev >= x % (20 = 'Thr Venta' de la app)
HOLD_DEVX = [15, 30, 60]
HORIZON = 70                      # sesiones de ventana hacia delante (cubre hold 60 naturales)
SIGNALS = ["prophet", "linreg", "sma100"]
PROTOCOLS = ["post_close", "pre_open"]
BIN_HORIZONS = [5, 10, 15, 20, 30]
REFIT_GUARD = [5]                 # se fija en run(); dev=NaN si el ajuste vigente es mas viejo que 2x refit


def _ffill(a):
    a = a.copy()
    mask = np.isnan(a)
    idx = np.where(~mask, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


# ---------------------------------------------------------------------------
# Señales
# ---------------------------------------------------------------------------
def build_signals(wide, cal, bank, meta):
    """Una fila por (ticker, dia D del calendario) con dev de cada señal y protocolo."""
    close = wide["close"].reindex(cal)
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    meta_ok = meta[meta["ok"]].sort_values("fit_date")
    rows = []
    for t in C.TICKERS:
        if t not in close.columns or t not in set(meta_ok["ticker"]):
            continue
        cl = close[t]
        df = pd.DataFrame({"D": cal, "close": cl.values})
        df["ticker"] = t
        df["close_prev"] = df["close"].shift(1)
        df["pos"] = np.arange(len(cal))
        df["D_prev"] = df["D"].shift(1)
        m = meta_ok[meta_ok["ticker"] == t][["fit_date", "r2", "sigma", "growth", "adr"]]
        b = bank[bank["ticker"] == t][["fit_date", "ds", "yhat"]]
        # post_close: F = ultimo fit <= D, yhat_F(D)
        x = pd.merge_asof(df.sort_values("D"), m, left_on="D", right_on="fit_date", direction="backward")
        x = x.merge(b, left_on=["fit_date", "D"], right_on=["fit_date", "ds"], how="left")
        x["k_post"] = x["pos"] - x["fit_date"].map(cal_pos)
        x["dev_post"] = (x["close"] / x["yhat"] - 1) * 100
        x.loc[~(x["k_post"] <= 2 * REFIT_GUARD[0]), "dev_post"] = np.nan
        x = x.rename(columns={"yhat": "yhat_post"}).drop(columns=["ds"])
        # pre_open: F = ultimo fit <= D-1, yhat_F(D), px = close(D-1)
        y = df[["D", "D_prev"]].dropna()
        y = pd.merge_asof(y.sort_values("D_prev"), m[["fit_date"]], left_on="D_prev", right_on="fit_date", direction="backward")
        y = y.merge(b, left_on=["fit_date", "D"], right_on=["fit_date", "ds"], how="left")
        y = y.rename(columns={"yhat": "yhat_pre", "fit_date": "fit_date_pre"})[["D", "yhat_pre", "fit_date_pre"]]
        # screening del protocolo pre_open: metricas del ajuste <= D-1 (no del que ya incluye el cierre de D)
        y = y.merge(m.rename(columns={"fit_date": "fit_date_pre", "r2": "r2_pre", "sigma": "sigma_pre",
                                      "growth": "growth_pre", "adr": "adr_pre"}), on="fit_date_pre", how="left")
        x = x.merge(y, on="D", how="left")
        x["k_pre"] = (x["pos"] - 1) - x["fit_date_pre"].map(cal_pos)
        x["dev_pre"] = (x["close_prev"] / x["yhat_pre"] - 1) * 100
        x.loc[~(x["k_pre"] <= 2 * REFIT_GUARD[0]), "dev_pre"] = np.nan
        # controles sin Prophet (in-sample en D, como post_close)
        lc = np.log(cl.values)
        n = len(lc)
        yh_lr = np.full(n, np.nan)
        W = 252
        for i in range(W - 1, n):
            seg = lc[i - W + 1:i + 1]
            ok = ~np.isnan(seg)
            if ok.sum() < C.MIN_TRAIN_ROWS:
                continue
            xs = np.arange(W)[ok]
            coef = np.polyfit(xs, seg[ok], 1)
            yh_lr[i] = np.exp(coef[0] * (W - 1) + coef[1])
        x["dev_linreg"] = (x["close"] / yh_lr - 1) * 100
        sma = pd.Series(cl.values).rolling(100, min_periods=90).mean().values
        x["dev_sma100"] = (x["close"] / sma - 1) * 100
        x["ret1d"] = x["close"] / x["close_prev"] - 1
        rows.append(x)
    sig = pd.concat(rows, ignore_index=True)
    sig["screen"] = [C.passes_screen(r, s, g, a) if pd.notna(r) else False
                     for r, s, g, a in zip(sig["r2"], sig["sigma"], sig["growth"], sig["adr"])]
    sig["screen_pre"] = [C.passes_screen(r, s, g, a) if pd.notna(r) else False
                         for r, s, g, a in zip(sig["r2_pre"], sig["sigma_pre"], sig["growth_pre"], sig["adr_pre"])]
    sig["screen_live"] = [C.passes_screen(r, s, g, a, C.SCREEN_LIVE) if pd.notna(r) else False
                          for r, s, g, a in zip(sig["r2"], sig["sigma"], sig["growth"], sig["adr"])]
    sig["screen_live_pre"] = [C.passes_screen(r, s, g, a, C.SCREEN_LIVE) if pd.notna(r) else False
                              for r, s, g, a in zip(sig["r2_pre"], sig["sigma_pre"], sig["growth_pre"], sig["adr_pre"])]
    sig["period"] = sig["D"].map(C.period_of)
    qqq = close["QQQ"]
    regime = (qqq > qqq.rolling(100).mean()).reindex(cal).values
    sig["qqq_above_sma100"] = sig["pos"].map(lambda p: bool(regime[p]) if p < len(regime) else False)
    sig["gap_proxy"] = sig["ret1d"].abs() > 0.06
    return sig


# ---------------------------------------------------------------------------
# Salidas precomputadas por ticker
# ---------------------------------------------------------------------------
class TickerPaths:
    """Para cada dia de fill e (indice en calendario): fill price y, por nivel, el indice de salida."""

    def __init__(self, t, wide, cal, fx, dev_post):
        self.t = t
        self.dates = cal.values
        self.n = len(cal)
        op = wide["open"][t].reindex(cal).values.astype(float)
        cl = wide["close"][t].reindex(cal).values.astype(float)
        self.open_raw = op
        self.open = _ffill(op)
        self.close = _ffill(cl)
        self.fx = fx
        self.fill = self.open * (1 + C.SLIP_BP / 1e4)
        H = HORIZON
        pad = np.full(H, np.nan)
        Cw = sliding_window_view(np.concatenate([self.close, pad]), H)[: self.n]   # Cw[e, h] = close[e+h]
        self._Cw = Cw
        sl_levels = sorted({s for s in SL if s is not None} | {kc["sl"] for kc in KEY_CONFIGS.values() if kc["sl"] is not None})
        tp_levels = sorted({p for p in TP if p is not None} | {kc["tp"] for kc in KEY_CONFIGS.values() if kc["tp"] is not None})
        self.sl_idx = {s: self._first_hit(Cw <= (self.fill * (1 - s / 100))[:, None]) for s in sl_levels}
        self.tp_idx = {p: self._first_hit(Cw >= (self.fill * (1 + p / 100))[:, None]) for p in tp_levels}
        dv = np.concatenate([dev_post, pad])
        Dw = sliding_window_view(dv, H)[: self.n]
        self.devx_idx = {x: self._first_hit(Dw >= x) for x in DEV_EXIT}
        # hold: la app cuenta dias naturales desde la entrada. Usuario de mañana (pre_open): ve days>=h la
        # mañana del primer dia de mercado >= fill+h y vende en esa apertura (side='left'). Usuario de tarde
        # (post_close): lo ve la tarde del dia natural fill+h y vende en la siguiente apertura (side='right').
        self.hold_idx = {"pre_open": {}, "post_close": {}}
        for h in set(HOLD + HOLD_DEVX + [kc["hold"] for kc in KEY_CONFIGS.values() if kc["hold"]]):
            target = self.dates + np.timedelta64(h, "D")
            self.hold_idx["pre_open"][h] = np.searchsorted(self.dates, target, side="left")
            self.hold_idx["post_close"][h] = np.searchsorted(self.dates, target, side="right")
        # MFE/MAE en 15 naturales (para diagnostico) sobre cierres
        self.Cw = Cw

    def _first_hit(self, hit):
        any_hit = hit.any(axis=1)
        h = hit.argmax(axis=1)
        trig = np.where(any_hit, np.arange(self.n) + h, self.n + 10 ** 6)
        return trig + 1          # venta a la apertura del dia siguiente al cierre que dispara

    def exit_for(self, sl, tp, hold, devx=None, proto="post_close"):
        cands = []
        if hold is not None:
            cands.append((self.hold_idx[proto][hold], 0))
        if sl is not None:
            cands.append((self.sl_idx[sl], 1))
        if tp is not None:
            cands.append((self.tp_idx[tp], 2))
        if devx is not None:
            cands.append((self.devx_idx[devx], 3))
        idx = np.stack([c[0] for c in cands], axis=1)
        reason_code = np.array([c[1] for c in cands])
        best = idx.argmin(axis=1)
        exit_idx = idx[np.arange(self.n), best]
        reason = reason_code[best]
        return exit_idx, reason

    def returns(self, e, exit_idx):
        """Retorno bruto USD y neto EUR de entrar en open[e] y salir en open[exit_idx] (o marcado al ultimo cierre)."""
        e = np.asarray(e)
        ex = np.asarray(exit_idx)
        open_end = ex >= self.n
        exi = np.minimum(ex, self.n - 1)
        exit_px = np.where(open_end, self.close[self.n - 1], self.open[exi] * (1 - C.SLIP_BP / 1e4))
        gross = exit_px / self.fill[e] - 1
        fx_in, fx_out = self.fx[e], self.fx[exi]
        net = (1 + gross) * (fx_in / fx_out) * (1 - 2 * C.FX_SPREAD_BP / 1e4) - 1 - 2 * C.FEE_EUR / C.SLOT_EUR
        days = (self.dates[exi] - self.dates[e]).astype("timedelta64[D]").astype(int)
        return gross, net, days, open_end

    def qqq_eur(self, qqq_open, e, exi):
        """QQQ en la misma ventana, en EUR (mismo EUR/USD que sufre la operacion)."""
        return (qqq_open[exi] / qqq_open[e]) * (self.fx[e] / self.fx[exi]) - 1


def sequential_trades(signal_e, exit_idx):
    """Una posicion por ticker: toma la primera señal, salta a la siguiente señal >= salida."""
    S = np.asarray(signal_e)
    out = []
    if len(S) == 0:
        return out
    i = 0
    while i < len(S):
        e = S[i]
        x = exit_idx[e]
        out.append(e)
        i = np.searchsorted(S, x, side="right")    # siguiente señal estrictamente despues de la salida (sin recompra el mismo dia)
    return out


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def exit_configs():
    cfgs = [dict(sl=s, tp=p, hold=h, devx=None) for s in SL for p in TP for h in HOLD]
    cfgs += [dict(sl=s, tp=None, hold=h, devx=x) for s in SL for x in DEV_EXIT for h in HOLD_DEVX]
    for kc in KEY_CONFIGS.values():   # p.ej. C_LIVE (SL 6 / TP 7) no esta en el grid
        xc = {k: kc[k] for k in ("sl", "tp", "hold", "devx")}
        if xc not in cfgs:
            cfgs.append(xc)
    return cfgs


def entry_configs():
    return [dict(entry=a, max_dev=b) for a in ENTRY for b in MAXDEV]


def cfg_key(ec, xc):
    return f"e{ec['entry']}_md{ec['max_dev']}_sl{xc['sl']}_tp{xc['tp']}_dx{xc['devx']}_h{xc['hold']}"


KEY_CONFIGS = {
    "C0":        dict(entry=3, max_dev=15, sl=15, tp=20, hold=15, devx=None),
    "C_LIVE":    dict(entry=2, max_dev=15, sl=6, tp=7, hold=15, devx=None),
    "HOLD15":    dict(entry=3, max_dev=15, sl=None, tp=None, hold=15, devx=None),
}


ENV = {}
MODES = [("prophet", "post_close", True), ("prophet", "pre_open", True), ("prophet", "post_close", False),
         ("prophet", "post_close", "live"), ("prophet_k0", "post_close", True),
         ("linreg", "post_close", False), ("sma100", "post_close", False)]


def _build_env(refit_every):
    """Precios, señales, rutas de salida y arrays por ticker (se construye en cada proceso)."""
    wide = C.load_prices()
    cal = C.trading_calendar(wide)
    sig = pd.read_parquet(os.path.join(C.RES_DIR, f"signals_refit{refit_every}.parquet"))
    fx = wide["close"]["EURUSD=X"].reindex(cal).ffill().bfill().values.astype(float)
    qqq_open = _ffill(wide["open"]["QQQ"].reindex(cal).values.astype(float))
    period_by_pos = np.array([C.period_of(d) for d in cal])
    period_names = [pp[0] for pp in C.PERIODS]
    period_code = {p: i for i, p in enumerate(period_names)}
    per_code_by_pos = np.array([period_code.get(p, -1) for p in period_by_pos])
    paths, A = {}, {}
    for t, g in sig.groupby("ticker"):
        g = g.sort_values("pos")
        dv = g.set_index("pos")["dev_post"].reindex(range(len(cal))).values.astype(float)
        P = TickerPaths(t, wide, cal, fx, dv)
        paths[t] = P
        A[t] = dict(
            pos=g["pos"].values,
            dev=dict(post_close={"prophet": g["dev_post"].values, "linreg": g["dev_linreg"].values,
                                 "sma100": g["dev_sma100"].values},
                     pre_open={"prophet": g["dev_pre"].values}),
            k=dict(post_close=g["k_post"].values, pre_open=g["k_pre"].values),
            screen={True: dict(post_close=g["screen"].values.astype(bool), pre_open=g["screen_pre"].values.astype(bool)),
                    "live": dict(post_close=g["screen_live"].values.astype(bool), pre_open=g["screen_live_pre"].values.astype(bool))},
            gap=g["gap_proxy"].values.astype(bool),
            regime=g["qqq_above_sma100"].values.astype(bool),
            open_ok=~np.isnan(P.open_raw),
        )
    return dict(sig=sig, paths=paths, A=A, qqq_open=qqq_open, period_by_pos=period_by_pos,
                period_names=period_names, per_code_by_pos=per_code_by_pos, cal_dates=cal.values)


def _init_worker(refit_every):
    import warnings as _w
    _w.filterwarnings("ignore")
    ENV.update(_build_env(refit_every))


def _process_exits(args):
    """Procesa una lista de configs de salida contra todas las entradas/modos. Devuelve (resumenes, trades clave)."""
    xcfgs, ecfgs, modes = args
    paths, A, qqq_open = ENV["paths"], ENV["A"], ENV["qqq_open"]
    period_names, per_code_by_pos, cal_dates = ENV["period_names"], ENV["per_code_by_pos"], ENV["cal_dates"]
    summaries, key_trades = [], []
    for xc in xcfgs:
        # salida + control incondicional por ticker/periodo (misma regla, entrada cualquier dia), por protocolo
        ex_by_t, unc_by_t = {}, {}
        for proto in PROTOCOLS:
            for t, P in paths.items():
                exit_idx, reason = P.exit_for(xc["sl"], xc["tp"], xc["hold"], xc["devx"], proto)
                # embargo: una operacion cuenta en el periodo de su fill solo si su salida por hold (la mas
                # tardia posible) cae dentro del mismo periodo; decidido por fill+hold, no por el precio
                hold_end = np.minimum(P.hold_idx[proto][xc["hold"]], P.n - 1)
                same_per = per_code_by_pos[hold_end] == per_code_by_pos
                e_all = np.arange(P.n - 1)
                g_, nt, d_, oe = P.returns(e_all, exit_idx[e_all])
                valid = A[t]["open_ok"][e_all] & ~oe & same_per[e_all]
                unc = np.full(len(period_names), np.nan)
                pc = per_code_by_pos[e_all]
                for i in range(len(period_names)):
                    m = valid & (pc == i)
                    if m.sum() > 20:
                        unc[i] = np.nanmean(nt[m])
                ex_by_t[(proto, t)] = (exit_idx, reason, same_per)
                unc_by_t[(proto, t)] = unc
        for sname, proto, scr in modes:
            for ec in ecfgs:
                if sname not in ("prophet", "prophet_k0") and ec["max_dev"] not in (15, 999):
                    continue
                if (sname == "prophet_k0" or scr == "live") and ec not in (dict(entry=3, max_dev=15), dict(entry=2, max_dev=15)):
                    continue
                cols = {k: [] for k in ["ticker", "e", "exi", "per", "reason", "open_end", "gross", "net",
                                        "days", "qqq", "uncond", "dev", "k", "gap", "regime", "purge"]}
                for t, a in A.items():
                    P = paths[t]
                    dev = a["dev"][proto]["prophet" if sname == "prophet_k0" else sname]
                    e = a["pos"] + (1 if proto == "post_close" else 0)
                    ok = (dev <= -ec["entry"]) & (dev > -ec["max_dev"]) & (e < P.n - 1)
                    ok &= a["open_ok"][np.minimum(e, P.n - 1)]
                    if scr:
                        ok &= a["screen"][scr][proto]
                    if sname == "prophet_k0":
                        ok &= a["k"][proto] == 0
                    if not ok.any():
                        continue
                    S = e[ok]
                    exit_idx, reason, same_per = ex_by_t[(proto, t)]
                    taken = sequential_trades(S, exit_idx)
                    if not taken:
                        continue
                    taken = np.array(taken)
                    sel = np.searchsorted(S, taken)          # indices en S -> indices en arrays de señal
                    sig_rows = np.flatnonzero(ok)[sel]
                    gr, nt, dd, oe = P.returns(taken, exit_idx[taken])
                    exi = np.minimum(exit_idx[taken], P.n - 1)
                    pc = per_code_by_pos[taken]
                    cols["ticker"].append(np.full(len(taken), t)); cols["e"].append(taken); cols["exi"].append(exi)
                    cols["per"].append(pc); cols["reason"].append(reason[taken]); cols["open_end"].append(oe)
                    cols["gross"].append(gr); cols["net"].append(nt); cols["days"].append(dd)
                    cols["qqq"].append(P.qqq_eur(qqq_open, taken, exi))
                    cols["uncond"].append(unc_by_t[(proto, t)][np.maximum(pc, 0)])
                    cols["dev"].append(dev[sig_rows]); cols["k"].append(a["k"][proto][sig_rows])
                    cols["gap"].append(a["gap"][sig_rows]); cols["regime"].append(a["regime"][sig_rows])
                    cols["purge"].append(~same_per[taken])
                if not cols["e"]:
                    continue
                tr = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
                tr["alpha"] = tr["net"] - tr["uncond"]
                tr["period"] = np.array(period_names + ["-"])[np.where(tr["per"] >= 0, tr["per"], len(period_names))]
                key = cfg_key(ec, xc)
                trc = tr[~tr["open_end"] & ~tr["purge"]]
                trc = trc.assign(pos_net=trc["net"].clip(lower=0), neg_net=(-trc["net"]).clip(lower=0),
                                 win=trc["net"] > 0, is_hold=trc["reason"] == 0, is_sl=trc["reason"] == 1,
                                 is_tp=trc["reason"] == 2, is_dx=trc["reason"] == 3, k0=trc["k"] == 0,
                                 qx=trc["net"] - trc["qqq"],
                                 a_gap=np.where(trc["gap"], trc["alpha"], np.nan),
                                 a_nogap=np.where(~trc["gap"], trc["alpha"], np.nan),
                                 a_on=np.where(trc["regime"], trc["alpha"], np.nan),
                                 a_off=np.where(~trc["regime"], trc["alpha"], np.nan),
                                 a_k0=np.where(trc["k"] == 0, trc["alpha"], np.nan))
                agg = trc.groupby("period").agg(
                    n_closed=("net", "size"), n_tickers=("ticker", "nunique"), mean_net=("net", "mean"),
                    median_net=("net", "median"), mean_gross=("gross", "mean"), mean_alpha=("alpha", "mean"),
                    mean_qqq_excess=("qx", "mean"), win_rate=("win", "mean"), pos_sum=("pos_net", "sum"),
                    neg_sum=("neg_net", "sum"), sh_hold=("is_hold", "mean"), sh_sl=("is_sl", "mean"),
                    sh_tp=("is_tp", "mean"), sh_devx=("is_dx", "mean"), mean_days=("days", "mean"),
                    sh_k0=("k0", "mean"), mean_alpha_k0=("a_k0", "mean"), mean_alpha_gap=("a_gap", "mean"),
                    mean_alpha_nogap=("a_nogap", "mean"), mean_alpha_regime_on=("a_on", "mean"),
                    mean_alpha_regime_off=("a_off", "mean"))
                agg["n"] = tr.groupby("period").size()
                agg["profit_factor"] = np.where(agg["neg_sum"] > 0, agg["pos_sum"] / agg["neg_sum"].replace(0, np.nan), np.inf)
                agg = agg.drop(columns=["pos_sum", "neg_sum"]).reset_index()
                agg["key"] = key; agg["signal"] = sname; agg["protocol"] = proto; agg["screen"] = str(scr)
                for f in ["entry", "max_dev"]:
                    agg[f] = ec[f]
                for f in ["sl", "tp", "devx", "hold"]:
                    agg[f] = xc[f]
                summaries.append(agg)
                for kname, kc in KEY_CONFIGS.items():
                    if all(kc[f] == ({**ec, **xc})[f] for f in ["entry", "max_dev", "sl", "tp", "hold", "devx"]):
                        tr2 = tr.copy()
                        tr2["fill_date"] = cal_dates[tr2["e"].values]; tr2["exit_date"] = cal_dates[tr2["exi"].values]
                        tr2["cfg"] = kname; tr2["signal"] = sname; tr2["protocol"] = proto; tr2["screen"] = str(scr)
                        key_trades.append(tr2)
    return (pd.concat(summaries, ignore_index=True) if summaries else None,
            pd.concat(key_trades, ignore_index=True) if key_trades else None)


def run(refit_every=5, quick=False, workers=8):
    t0 = time.time()
    REFIT_GUARD[0] = refit_every
    wide = C.load_prices()
    cal = C.trading_calendar(wide)
    bank, meta = load_bank(refit_every)
    print(f"[level1] banco: {bank['ticker'].nunique()} tickers, {meta['ok'].sum()} ajustes ok", flush=True)
    sig = build_signals(wide, cal, bank, meta)
    sig_path = os.path.join(C.RES_DIR, f"signals_refit{refit_every}.parquet")
    sig.to_parquet(sig_path, index=False)
    print(f"[level1] señales: {len(sig)} filas ticker-dia -> {sig_path} ({time.time()-t0:.0f}s)", flush=True)

    xcfgs = exit_configs()
    ecfgs = entry_configs()
    if quick:
        keys = [{k: v for k, v in kc.items() if k in ("sl", "tp", "hold", "devx")} for kc in KEY_CONFIGS.values()]
        xcfgs = [x for x in xcfgs if x in keys] + [dict(sl=10, tp=None, hold=15, devx=0)]
        ecfgs = [dict(entry=3, max_dev=15), dict(entry=2, max_dev=15)]
    n_chunks = max(1, min(workers * 4, len(xcfgs)))
    chunks = [(xcfgs[i::n_chunks], ecfgs, MODES) for i in range(n_chunks)]
    print(f"[level1] {len(xcfgs)} configs de salida x {len(ecfgs)} de entrada x {len(MODES)} modos en {workers} procesos", flush=True)
    summaries, key_trades = [], []
    if workers <= 1 or quick:
        _init_worker(refit_every)
        results = [_process_exits(c) for c in chunks]
    else:
        from multiprocessing import Pool
        with Pool(workers, initializer=_init_worker, initargs=(refit_every,)) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_process_exits, chunks)):
                results.append(r)
                print(f"  bloque {i+1}/{n_chunks}  {time.time()-t0:.0f}s", flush=True)
    for s, k in results:
        if s is not None:
            summaries.append(s)
        if k is not None:
            key_trades.append(k)
    summ = pd.concat(summaries, ignore_index=True)
    summ.to_parquet(os.path.join(C.RES_DIR, f"level1_summary_refit{refit_every}.parquet"), index=False)
    kt = pd.concat(key_trades, ignore_index=True)
    kt.to_parquet(os.path.join(C.RES_DIR, f"level1_keytrades_refit{refit_every}.parquet"), index=False)
    print(f"[level1] resumen: {len(summ)} filas (combo x modo x periodo); trades clave: {len(kt)}  ({time.time()-t0:.0f}s)", flush=True)

    env = ENV if ENV else _build_env(refit_every)
    diagnostics(env["sig"], env["paths"], env["qqq_open"], env["period_by_pos"], refit_every)
    print(f"[level1] listo en {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Diagnosticos sin grid: curva por bin de desviacion y MAE/MFE
# ---------------------------------------------------------------------------
def diagnostics(sig, paths, qqq_open, period_by_pos, refit_every):
    rows = []
    bins = list(range(-25, 1))
    for sname in SIGNALS:
        col = "dev_post" if sname == "prophet" else f"dev_{sname}"
        for t, g in sig.groupby("ticker"):
            P = paths[t]
            e = g["pos"].values + 1
            ok = (e < P.n - HORIZON) & ~np.isnan(g[col].values) & ~np.isnan(P.open_raw[np.minimum(e, P.n - 1)])
            if sname == "prophet":
                ok &= g["screen"].values
            e = e[ok]; dev = g[col].values[ok]
            if len(e) == 0:
                continue
            per = period_by_pos[e]
            for h in BIN_HORIZONS:
                fwd = P.open[e + h] / P.fill[e] - 1
                qq = qqq_open[e + h] / qqq_open[e] - 1
                # control incondicional del horizonte: media de fwd de todos los dias del ticker/periodo
                e_all = np.arange(P.n - HORIZON)
                fwd_all = P.open[e_all + h] / P.fill[e_all] - 1
                per_all = period_by_pos[e_all]
                unc = {p: np.nanmean(fwd_all[(per_all == p) & ~np.isnan(P.open_raw[e_all])]) for p in set(per)}
                rows.append(pd.DataFrame({"signal": sname, "ticker": t, "e": e, "period": per, "h": h,
                                          "dev": dev, "fwd": fwd, "qqq": qq,
                                          "alpha": fwd - np.array([unc[p] for p in per])}))
    db = pd.concat(rows, ignore_index=True)
    db["bin"] = pd.cut(db["dev"], bins=[-100] + bins + [100], labels=False) - 1
    db = db[(db["dev"] <= 0) & (db["dev"] > -25)]
    db["bin"] = np.floor(db["dev"]).astype(int)
    agg = db.groupby(["signal", "period", "h", "bin"]).agg(n=("alpha", "size"), alpha=("alpha", "mean"),
                                                            fwd=("fwd", "mean"), qqq=("qqq", "mean")).reset_index()
    agg.to_parquet(os.path.join(C.RES_DIR, f"level1_devbins_refit{refit_every}.parquet"), index=False)
    # MAE/MFE de la señal base (entry 3, max_dev 15) a 15 naturales, sin SL/TP
    mm = []
    for t, g in sig.groupby("ticker"):
        P = paths[t]
        e = g["pos"].values + 1
        dev = g["dev_post"].values
        ok = (dev <= -3) & (dev > -15) & ~np.isnan(dev) & g["screen"].values.astype(bool) & (e < P.n - HORIZON)
        S = sequential_trades(e[ok], P.hold_idx["post_close"][15])
        for s in S:
            x = min(P.hold_idx["post_close"][15][s], P.n - 1)
            path = P.close[s:x] / P.fill[s] - 1
            if len(path) == 0:
                continue
            mm.append(dict(ticker=t, e=s, period=period_by_pos[s], mae=path.min(), mfe=path.max(),
                           final=P.open[x] / P.fill[s] - 1))
    pd.DataFrame(mm).to_parquet(os.path.join(C.RES_DIR, f"level1_maemfe_refit{refit_every}.parquet"), index=False)
    print(f"[level1] diagnosticos: {len(agg)} bins, {len(mm)} trades MAE/MFE", flush=True)
