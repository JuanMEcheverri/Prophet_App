"""Backtest 15 - walk-forward honesto (ver wf15/ y README.md).

Uso:
  python 15_walkforward.py fit        [--refit 5] [--workers 8] [--tickers AAPL,MSFT] [--limit-dates N]
  python 15_walkforward.py level1     [--refit 5]
  python 15_walkforward.py universe   [--refit 5]
  python 15_walkforward.py portfolio  [--refit 5]
  python 15_walkforward.py report     [--refit 5]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "level1", "select", "universe", "portfolio", "report", "paper"])
    ap.add_argument("--refit", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tickers", type=str, default=None)
    ap.add_argument("--limit-dates", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--start", type=str, default=None, help="primer fit_date (YYYY-MM-DD); por defecto SIM_START")
    ap.add_argument("--refresh-prices", action="store_true")
    a = ap.parse_args()
    tickers = a.tickers.split(",") if a.tickers else None

    if a.stage == "fit":
        from wf15 import fit_bank
        fit_bank.run(refit_every=a.refit, workers=a.workers, tickers=tickers, limit_dates=a.limit_dates,
                     start=a.start, force_prices=a.refresh_prices)
    elif a.stage == "level1":
        from wf15 import level1
        level1.run(refit_every=a.refit, quick=a.quick, workers=a.workers)
    elif a.stage == "select":
        from wf15 import report
        report.select(refit_every=a.refit)
    elif a.stage == "universe":
        from wf15 import universe
        universe.run(refit_every=a.refit)
    elif a.stage == "portfolio":
        from wf15 import portfolio
        portfolio.run(refit_every=a.refit)
    elif a.stage == "report":
        from wf15 import report
        report.run(refit_every=a.refit)
    elif a.stage == "paper":
        from wf15 import paper_gen
        paper_gen.run(start=a.start or "2026-10-01", refit_every=a.refit)
