"""Banco de ajustes Prophet en calendario fijo.

Para cada ticker y cada fit_date F (cada REFIT_EVERY sesiones desde SIM_START):
  train = cierres en (F - 1 año, F]  (incluye el cierre de F, igual que la app
          cuando se entrena despues del cierre)
  Se guarda yhat para ds en [F, F + STORE_FUTURE_DAYS] (naturales) y las metricas
  de screening (r2, sigma, growth, adr) calculadas como en app.train_models().
Nunca entra en el ajuste ningun precio posterior a F (assert explicito).
Reanudable: un parquet por ticker; si existe con todas las fechas, se salta.
"""
import os, sys, time, warnings, logging
import numpy as np
import pandas as pd

from . import common as C


def _fit_one(train, high, low, fit_date):
    from prophet import Prophet
    df_fit = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(**C.PROPHET_KW)
    m.fit(df_fit)
    future = m.make_future_dataframe(periods=C.FUTURE_DAYS)
    pred = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    pred["ds"] = pred["ds"].dt.normalize()
    pred = pred.set_index("ds")
    r2, sigma, growth, adr = C.screening_metrics(train, high, low, pred)
    keep = pred[(pred.index >= fit_date) & (pred.index <= fit_date + pd.Timedelta(days=C.STORE_FUTURE_DAYS))]
    return keep, (r2, sigma, growth, adr)


def worker(args):
    ticker, dates, close, high, low = args
    warnings.filterwarnings("ignore")
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    rows, meta = [], []
    t0 = time.time()
    for F in dates:
        train = close[(close.index > F - C.TRAIN_WINDOW) & (close.index <= F)].dropna()
        if len(train) < C.MIN_TRAIN_ROWS:
            meta.append(dict(ticker=ticker, fit_date=F, n_train=len(train), ok=False,
                             r2=np.nan, sigma=np.nan, growth=np.nan, adr=np.nan))
            continue
        assert train.index.max() <= F
        try:
            keep, (r2, sigma, growth, adr) = _fit_one(train, high, low, F)
        except Exception as e:
            meta.append(dict(ticker=ticker, fit_date=F, n_train=len(train), ok=False,
                             r2=np.nan, sigma=np.nan, growth=np.nan, adr=np.nan, err=str(e)[:80]))
            continue
        k = keep.reset_index()
        k["fit_date"] = F
        k["ticker"] = ticker
        rows.append(k)
        meta.append(dict(ticker=ticker, fit_date=F, n_train=len(train), ok=True,
                         r2=r2, sigma=sigma, growth=growth, adr=adr))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["ds", "yhat", "yhat_lower", "yhat_upper", "fit_date", "ticker"])
    return ticker, out, pd.DataFrame(meta), time.time() - t0


def run(refit_every=5, workers=8, tickers=None, limit_dates=None):
    wide = C.load_prices()
    cal = C.trading_calendar(wide)
    dates = C.fit_dates(cal, refit_every)
    if limit_dates:
        dates = dates[:limit_dates]
    out_dir = C.fit_bank_dir(refit_every)
    tickers = tickers or C.TICKERS
    todo = []
    for t in tickers:
        p = os.path.join(out_dir, f"{t}.parquet")
        pm = os.path.join(out_dir, "meta", f"{t}.parquet")
        if os.path.exists(p) and os.path.exists(pm) and not limit_dates:
            done = pd.read_parquet(pm)
            if len(done) >= len(dates):
                continue
        close = wide["close"][t].dropna()
        todo.append((t, list(dates), close, wide["high"][t].reindex(close.index), wide["low"][t].reindex(close.index)))
    print(f"[fit_bank] refit cada {refit_every} sesiones: {len(dates)} fechas x {len(tickers)} tickers; "
          f"pendientes {len(todo)} tickers (~{len(todo)*len(dates)} ajustes), workers={workers}", flush=True)
    if not todo:
        return
    t0 = time.time()
    done_n = 0
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for ticker, out, meta, secs in pool.imap_unordered(worker, todo):
            for c in ["r2", "sigma", "growth", "adr"]:
                meta[c] = pd.to_numeric(meta[c], errors="coerce").astype(float)
            meta.to_parquet(os.path.join(out_dir, "meta", f"{ticker}.parquet"), index=False)
            out.to_parquet(os.path.join(out_dir, f"{ticker}.parquet"), index=False)
            done_n += 1
            el = time.time() - t0
            eta = el / done_n * (len(todo) - done_n)
            print(f"  {ticker:6s} {len(meta):4d} fechas  ok={int(meta['ok'].sum()):4d}  {secs:6.1f}s  "
                  f"[{done_n}/{len(todo)}  {el/60:.1f} min, ETA {eta/60:.1f} min]", flush=True)
    print(f"[fit_bank] listo en {(time.time()-t0)/60:.1f} min", flush=True)


def load_bank(refit_every=5):
    """Devuelve (yhat_long, meta) del banco completo."""
    d = C.fit_bank_dir(refit_every)
    parts, metas = [], []
    for t in C.TICKERS:
        p = os.path.join(d, f"{t}.parquet")
        pm = os.path.join(d, "meta", f"{t}.parquet")
        if os.path.exists(p) and os.path.exists(pm):
            parts.append(pd.read_parquet(p))
            metas.append(pd.read_parquet(pm))
    yh = pd.concat(parts, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    return yh, meta
