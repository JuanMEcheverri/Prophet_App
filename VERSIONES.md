# Versiones de Prophet Trader

Registro de lo que se sube a `master` (y por tanto a laprofecia.streamlit.app). La versión más reciente va arriba.

| Versión | Fecha | Commit | Resumen |
|---|---|---|---|
| **2.5** | 2026-11-04 | — | La barra de pestañas pasa a dos filas cuando no cabe (antes se recortaba); la pestaña activa se marca con un borde azul. Reaplicado el screening revisado, que un "Restaurar" desde la versión anterior había devuelto al perfil antiguo. |
| 2.4 | 2026-11-04 | `9ee67f5` | Pestaña "ℹ️ Versiones" en la app que muestra este registro, y número de versión en la barra lateral. |
| 2.3 | 2026-11-04 | `9d578b2` | Screening revisado con 5 años de datos: solo ADR ≤ 4 % y crecimiento ≤ 20 %, sin mínimo de crecimiento (R² y sigma desactivados). Pasan 36 de 87 acciones en vez de 21. El slider de crecimiento mínimo admite negativos. Cartera de papel regenerada con el nuevo perfil. |
| 2.2 | 2026-11-04 | `b47caa4` | Dos pestañas nuevas "🧪 Claude · Backtest 15" y "🧪 Claude · Combinado": cartera de papel de 1.000 $ en 3 slots que decide cada día solo con los datos de ese día y ejecuta a la apertura siguiente; incluye un mes de histórico (oct-nov 2026). En "Mis Posiciones": registro de salidas, precios en EUR o USD convertidos con el EUR/USD del día, tabla de ventas con P&L realizado. |
| 2.1 | 2026-11-04 | `fc5c47c`, `4b877f3` | Los JSON persistidos (estrategia, prophet, screening, tickers) pasan a la configuración del backtest 15 para que la nube la use sin pulsar nada; universo completo (87 tickers que cotizan). |
| 2.0 | 2026-11-04 | `20a47c1` | Backtest 15 (`backtests/wf15/`): walk-forward honesto 2021-2026 con fills a la apertura siguiente, comisiones, EUR/USD, controles y comparación con QQQ. Conclusión: la config antigua (3 / TP 20 / SL 15 / 15 d) perdía contra QQQ en 4 de 5 años; la lista de 55 no se distingue del azar. Nuevos defaults: entrada 4 % bajo yhat, salida al volver a yhat +7 % (anclada a yhat también en Posiciones), SL 7 %, 60 días, ventana de 1 año, mínimo 200 cierres. Nota al principio del README de backtests. |
| 1.2 | 2026-11-04 | `a4919b2` | Botón en la barra lateral para restaurar la configuración validada por backtest (entonces la de los backtests 07-14 y la lista de 55). |
| 1.1 | 2026-08-01 | `5e8a305` | Config "ganadora" de los backtests 07-14 (SL 15 / TP 20 / 15 d / max_dev 15 / 3 slots) y archivo de los 14 scripts de backtest con README. |
| 1.0 | 2026-07 | — | App inicial: Prophet por acción, señales, screening, charts, proyecciones, posiciones, persistencia en GitHub desde Streamlit Cloud. |

## Cómo añadir una versión

Cada vez que se suba algo a `master`: nueva fila arriba con número (mayor.menor), fecha, commit corto (`git log -1 --oneline`) y dos o tres frases de qué cambia para el usuario.
