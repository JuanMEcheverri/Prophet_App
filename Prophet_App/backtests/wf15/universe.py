"""Test de exclusion de tickers sin mirar el futuro.

En cada frontera S (1 ene / 1 jul) se construye la lista de exclusion SOLO con operaciones
cerradas antes de S, y se evalua en el semestre siguiente con las operaciones de nivel 1 de la
config C0 (prophet, post_close, screening ON). Reglas:
  R0   ninguna exclusion
  R1   regla del script 07 hacia delante: excluir si n>=3 y media P&L neto < 0 (12 meses previos)
  R1b  igual, con todo el historico previo
  R2   estadistica: excluir si n>=8 y el IC 90% (bootstrap) de la media de alpha < 0 (24 meses)
  R3   relativa: excluir el cuartil inferior por alpha media (12 meses)
  R4   sin P&L: excluir el cuartil superior de volatilidad realizada 90d a fecha S
  L07  la lista fija de 55 del script 07 (referencia, con look-ahead)
  PLACEBO  200 listas aleatorias del tamaño de R1
Las reglas y la persistencia usan solo neto y exceso vs QQQ de la misma ventana (conocibles en S);
NO la alpha de nivel 1, cuyo control incondicional promedia todo el periodo (incluiria el futuro).
Valor añadido = media neta de las operaciones de los tickers incluidos - media de todos (R0),
y lo mismo con exceso vs QQQ; ademas el B&H equiponderado de la lista incluida vs todos (si la regla
solo elige ganadores recientes, su B&H sube igual que su estrategia). Persistencia: Spearman de
la alpha media por ticker 12m previos vs 6m siguientes, y tasa 'tercil inferior sigue inferior'.
"""
import os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import common as C

BOUNDARIES = [pd.Timestamp(d) for d in ["2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01",
                                         "2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01"]]
N_PLACEBO = 200


def _boot_ci(x, n=500, q=(5, 95), seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, q[0]), np.percentile(m, q[1])


def run(refit_every=5, cfg_name="C0"):
    t0 = time.time()
    wide = C.load_prices()
    cal = C.trading_calendar(wide)
    close = wide["close"].reindex(cal)
    kt = pd.read_parquet(os.path.join(C.RES_DIR, f"level1_keytrades_refit{refit_every}.parquet"))
    tr = kt[(kt["cfg"] == cfg_name) & (kt["signal"] == "prophet") & (kt["protocol"] == "post_close") & (kt["screen"].astype(str) == "True")].copy()
    tr = tr[~tr["open_end"]].copy()
    tr["fill_date"] = pd.to_datetime(tr["fill_date"]); tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    # 'alpha' de nivel 1 resta la media incondicional de TODO el periodo (incluye el semestre futuro):
    # para decidir exclusiones solo se puede usar informacion conocida en S -> exceso vs QQQ en la misma ventana.
    tr["alpha_l1"] = tr["alpha"]
    tr["alpha"] = tr["net"] - tr["qqq"]
    logret = np.log(close[C.TICKERS]).diff()
    rows, pers = [], []
    rng = np.random.default_rng(42)
    for S in BOUNDARIES:
        S_end = S + pd.DateOffset(months=6)
        fwd = tr[(tr["fill_date"] >= S) & (tr["fill_date"] < S_end)]
        if len(fwd) < 20:
            continue
        past = tr[tr["exit_date"] < S]
        p12 = past[past["fill_date"] >= S - pd.DateOffset(months=12)]
        p24 = past[past["fill_date"] >= S - pd.DateOffset(months=24)]
        g12 = p12.groupby("ticker").agg(n=("net", "size"), net=("net", "mean"), alpha=("alpha", "mean"))
        gall = past.groupby("ticker").agg(n=("net", "size"), net=("net", "mean"))
        g24 = p24.groupby("ticker")["alpha"]
        ci_lo = {t: _boot_ci(v.values)[1] for t, v in g24}   # cota superior del IC 90%
        n24 = g24.size()
        vol = logret[(logret.index < S) & (logret.index >= S - pd.Timedelta(days=130))].std() * np.sqrt(252)
        rules = {
            "R0": set(),
            "R1": set(g12[(g12["n"] >= 3) & (g12["net"] < 0)].index),
            "R1b": set(gall[(gall["n"] >= 3) & (gall["net"] < 0)].index),
            "R2": {t for t in n24.index if n24[t] >= 8 and ci_lo.get(t, np.nan) < 0},
            "R3": set(g12[g12["n"] >= 2].sort_values("alpha").head(max(1, int(len(g12[g12["n"] >= 2]) * 0.25))).index),
            "R4": set(vol.dropna().sort_values(ascending=False).head(int(len(vol.dropna()) * 0.25)).index),
            "L07": set(C.TICKERS) - set(C.LIST_07),
        }
        # B&H equiponderado del semestre por ticker (para detectar 'momentum screen')
        w = close[(close.index >= S) & (close.index < S_end)]
        bh = (w[C.TICKERS].ffill().iloc[-1] / w[C.TICKERS].bfill().iloc[0] - 1)
        all_net, all_alpha = fwd["net"].mean(), fwd["alpha"].mean()
        placebo = []
        n_excl = len(rules["R1"])
        for k in range(N_PLACEBO if n_excl > 0 else 0):
            ex = set(rng.choice(C.TICKERS, size=n_excl, replace=False))
            inc = fwd[~fwd["ticker"].isin(ex)]
            placebo.append(inc["net"].mean() - all_net)
        p95 = np.percentile(placebo, 95) if placebo else np.nan
        for rname, ex in rules.items():
            inc = fwd[~fwd["ticker"].isin(ex)]
            exc = fwd[fwd["ticker"].isin(ex)]
            rows.append(dict(
                S=S, rule=rname, n_excluded=len(ex), n_trades_inc=len(inc), n_trades_exc=len(exc),
                net_inc=inc["net"].mean(), net_exc=exc["net"].mean() if len(exc) else np.nan,
                alpha_inc=inc["alpha"].mean(), alpha_exc=exc["alpha"].mean() if len(exc) else np.nan,
                value_added_net=inc["net"].mean() - all_net, value_added_alpha=inc["alpha"].mean() - all_alpha,
                bh_inc=bh[[t for t in C.TICKERS if t not in ex]].mean(), bh_all=bh.mean(),
                placebo_p95=p95 if rname == "R1" else np.nan,
                placebo_pct=(np.mean(np.array(placebo) <= inc["net"].mean() - all_net) if placebo else np.nan),
            ))
        # persistencia 12m previos -> 6m siguientes
        gf = fwd.groupby("ticker")["alpha"].mean()
        both = g12[g12["n"] >= 2].join(gf.rename("fwd"), how="inner")
        if len(both) >= 10:
            rho, p = spearmanr(both["alpha"], both["fwd"])
            t_lo = both["alpha"] <= both["alpha"].quantile(1 / 3)
            f_lo = both["fwd"] <= both["fwd"].quantile(1 / 3)
            pers.append(dict(S=S, n=len(both), spearman=rho, pval=p, bottom_stays_bottom=(t_lo & f_lo).sum() / max(1, t_lo.sum())))
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(C.RES_DIR, f"universe_refit{refit_every}.parquet"), index=False)
    pers_df = pd.DataFrame(pers)
    pers_df.to_parquet(os.path.join(C.RES_DIR, f"universe_persistence_refit{refit_every}.parquet"), index=False)
    print(f"[universe] {len(out)} filas regla x frontera, {len(pers_df)} fronteras con persistencia ({time.time()-t0:.0f}s)", flush=True)
    return out, pers_df
