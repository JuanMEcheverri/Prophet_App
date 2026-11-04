"""Backtest 15: walk-forward honesto de la estrategia Prophet Trader.

Modulos:
  common     - configuracion, descarga/cache de precios, calendario, metricas de screening
  fit_bank   - banco de ajustes Prophet en calendario fijo (la unica etapa cara)
  level1     - operaciones independientes sobre el banco: grid de reglas, controles, diagnosticos
  universe   - test de exclusion de tickers sin mirar el futuro
  portfolio  - motor de slots (capital limitado) sobre configs preseleccionadas
  report     - tablas y veredicto
"""
