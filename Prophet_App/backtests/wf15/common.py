import os, json, hashlib, warnings, logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

BT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(BT_DIR, "results", "15")
FITS_DIR = os.path.join(RES_DIR, "fits")
PRICES_PATH = os.path.join(RES_DIR, "prices.parquet")
os.makedirs(RES_DIR, exist_ok=True)

# Nasdaq-100 de app.py (90 unicos; MAR estaba duplicado) menos EA, WBD, ANSS (ya no cotizan en Yahoo)
TICKERS = sorted(set([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","TMUS","ADBE","CSCO","PEP","QCOM","INTU","AMAT",
    "TXN","ISRG","AMGN","BKNG","MU","VRTX","REGN","PANW","KLAC","GILD",
    "LRCX","ADI","SBUX","MDLZ","MELI","SNPS","CDNS","CTAS","PAYX","ROST",
    "DDOG","ABNB","PYPL","MAR","ORLY","FAST","IDXX","WDAY","PCAR","AEP",
    "FTNT","CPRT","MNST","BIIB","DLTR","TTD","DXCM","EBAY","ENPH",
    "XEL","EXC","ODFL","CRWD","NXPI","ON","ZS","VRSK","GEHC","HON",
    "FSLR","CEG","GFS","PDD","NTAP","APP","SMCI","ZBRA","KHC","ILMN",
    "OKTA","BMRN","MDB","ZM","SWKS","TEAM","MRVL","ROP",
]))
assert len(TICKERS) == 87, len(TICKERS)

# Lista de 55 del script 07 (solo como fila de referencia, nunca como universo)
LIST_07 = sorted([
    "ADBE","ADI","AEP","AMAT","AMD","AMGN","AMZN","AVGO","CDNS","CEG","COST",
    "CSCO","CTAS","DDOG","DLTR","DXCM","EXC","FAST","FSLR","FTNT","GEHC","GFS",
    "GILD","GOOG","GOOGL","IDXX","ISRG","KHC","KLAC","LRCX","MAR","MELI","META",
    "MNST","MRVL","MU","NTAP","NVDA","NXPI","ON","PANW","PAYX","PCAR","PDD",
    "QCOM","REGN","SNPS","TSLA","TTD","TXN","VRTX","XEL","ZBRA","ZM","ZS",
])

BENCH_TICKERS = ["QQQ", "EURUSD=X"]
DOWNLOAD_START = "2020-06-01"
SIM_START = pd.Timestamp("2021-08-02")       # primer dia de ajuste; necesita 1 año de historia antes
TRAIN_WINDOW = pd.DateOffset(years=1)
MIN_TRAIN_ROWS = 200                          # la app exige 250; con ventana de 1 año hay ~251 y los festivos lo rompen
FUTURE_DAYS = 43                              # app: fc_days(28) + 15 -> rango sobre el que calcula growth
STORE_FUTURE_DAYS = 15                        # dias naturales de yhat guardados a partir del fit_date

PROPHET_KW = dict(
    interval_width=0.95, seasonality_mode="multiplicative", growth="linear",
    changepoint_prior_scale=0.03, daily_seasonality=False,
    yearly_seasonality=True, weekly_seasonality=False,
)

# Periodos de evaluacion: años ago->jul, no solapados. P5 es el año en el que se
# eligieron todos los parametros de los scripts 07-14 (en muestra para C0 y LIST_07).
PERIODS = [
    ("P1_2021-22", "2021-08-02", "2022-07-31"),   # bajista (QQQ -13%)
    ("P2_2022-23", "2022-08-01", "2023-07-31"),
    ("P3_2023-24", "2023-08-01", "2024-07-31"),
    ("P4_2024-25", "2024-08-01", "2025-07-31"),
    ("P5_2025-26", "2025-08-01", "2026-07-31"),   # EN MUESTRA para la config actual y la lista de 55
    ("HOLD_2026",  "2026-08-01", "2099-12-31"),   # nunca visto por ningun script anterior
]

# Config actual de la app (C0) y config que esta en vivo en estrategia.json
C0 = dict(entry=3, max_dev=15, sl=15, tp=20, hold=15)
C_LIVE = dict(entry=2, max_dev=15, sl=6, tp=7, hold=15)

SCREEN_DEFAULTS = dict(min_r2=0.68, max_sigma=0.06, min_growth=0.02, max_growth=0.35, max_adr=0.04)
SCREEN_LIVE = dict(min_r2=0.68, max_sigma=0.06, min_growth=0.02, max_growth=0.49, max_adr=0.07)   # screening.json en vivo

# Costes
FEE_EUR = 1.0            # por operacion (compra o venta), como CAPITAL_DEFAULTS
SLIP_BP = 5.0            # deslizamiento por lado sobre la apertura
FX_SPREAD_BP = 10.0      # por conversion EUR<->USD
SLOT_EUR = 1000.0 / 3


def period_of(ts):
    for name, a, b in PERIODS:
        if pd.Timestamp(a) <= ts <= pd.Timestamp(b):
            return name
    return None


def load_prices(force=False):
    """Descarga una vez OHLC de los 86 + QQQ + EURUSD y lo cachea en parquet (formato largo)."""
    if os.path.exists(PRICES_PATH) and not force:
        long = pd.read_parquet(PRICES_PATH)
    else:
        import yfinance as yf
        raw = yf.download(TICKERS + BENCH_TICKERS, start=DOWNLOAD_START, auto_adjust=True, progress=False)
        frames = []
        for t in TICKERS + BENCH_TICKERS:
            try:
                df = pd.DataFrame({
                    "open": raw["Open"][t], "high": raw["High"][t],
                    "low": raw["Low"][t], "close": raw["Close"][t],
                }).dropna(subset=["close"])
            except KeyError:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            df["ticker"] = t
            df["date"] = df.index
            frames.append(df.reset_index(drop=True))
        long = pd.concat(frames, ignore_index=True)
        long["downloaded_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        long.to_parquet(PRICES_PATH, index=False)
    wide = {f: long.pivot(index="date", columns="ticker", values=f).sort_index() for f in ["open", "high", "low", "close"]}
    return wide


def trading_calendar(wide):
    """Calendario = dias con cierre de QQQ (NYSE)."""
    return wide["close"]["QQQ"].dropna().index


def fit_dates(calendar, refit_every, start=SIM_START):
    cal = calendar[calendar >= start]
    return cal[::refit_every]


def fit_bank_dir(refit_every):
    d = os.path.join(FITS_DIR, f"refit{refit_every}")
    os.makedirs(os.path.join(d, "meta"), exist_ok=True)
    return d


def screening_metrics(train, high, low, pred):
    """Replica train_models() de app.py / lineas 152-165 del script 14 sobre un ajuste."""
    yh = pred["yhat"]
    common = train.index.intersection(yh.index)
    y_is = train.loc[common].values
    yh_is = yh.loc[common].values
    resids = (y_is - yh_is) / np.clip(yh_is, 0.01, None)
    r2 = float(np.corrcoef(y_is, yh_is)[0, 1] ** 2) if len(common) > 1 else 0.0
    sigma = float(np.std(resids))
    y0, y1 = float(yh.iloc[0]), float(yh.iloc[-1])
    nyrs = (yh.index[-1] - yh.index[0]).days / 365.25
    if nyrs > 0 and y0 > 0 and y1 > 0:
        growth = (y1 / y0) ** (1 / nyrs) - 1
    else:
        growth = -1.0     # yhat extrapolado <= 0: en la app esto revienta el try/except y el ticker no pasa
    tail_n = min(90, len(train))
    cl_t = train.tail(tail_n)
    hi_t = high.reindex(cl_t.index)
    lo_t = low.reindex(cl_t.index)
    adr = float(((hi_t - lo_t) / cl_t.clip(lower=0.01)).mean())
    return r2, sigma, growth, adr


def passes_screen(r2, sigma, growth, adr, s=SCREEN_DEFAULTS):
    return (r2 >= s["min_r2"] and sigma <= s["max_sigma"]
            and s["min_growth"] <= growth <= s["max_growth"] and adr <= s["max_adr"])


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
