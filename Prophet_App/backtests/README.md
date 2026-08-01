# Backtests de Prophet Trader

Este directorio contiene todos los scripts de backtesting usados para afinar
`Prophet_App/app.py`: la ventana de entrenamiento de Prophet, sus
hiperparámetros, las reglas de entrada/salida de la estrategia, el screening,
y una simulación de portafolio con capital limitado y reinversión.

Todos los scripts son independientes de la app (no la importan), pero usan
**exactamente los mismos hiperparámetros y lógica de señales** que
`Prophet_App/app.py`, para que las conclusiones sean aplicables directamente.

## Resumen ejecutivo — configuración final

| Sección de la app | Parámetro | Valor final | Por qué |
|---|---|---|---|
| Prophet | Ventana de entrenamiento | **1 año** | Menor error de predicción y mejor P&L que 6 meses / 2 / 3 / 4 años (backtests 1-3) |
| Prophet | `changepoint_prior_scale` | **0.03** | Gana en mediana, media y tasa de outliers frente al 0.05 original (backtest 4-6) |
| Prophet | `seasonality_mode` | **multiplicative** (sin cambio) | "additive" mejora la mediana pero tiene cola de riesgo grave (media de error hasta 37% en el peor caso) |
| Estrategia | `entry_dev` | 3% (sin cambio) | Efecto marginal chico en el rango probado |
| Estrategia | `sl_pct` (Stop Loss) | **15%** | Un SL muy ajustado (2-3%) saca de la operación por ruido antes de que la reversión ocurra |
| Estrategia | `exit_dev`/TP (Take Profit) | **20%**, anclado al **precio de entrada** (no a yhat) | Cambio de diseño explícito — más simple y más fácil de razonar que un TP que se mueve con yhat |
| Estrategia | `max_hold` | **15 días** | Ganador claro sobre 20/30 días en todas las métricas |
| Estrategia | `max_dev` (cap caída máxima) | **15%** | Evita comprar en caídas extremas (ej. -52% bajo yhat) que suelen ser "cuchillos cayendo", no ruido |
| Screening | `min_r2` | 0.68 (sin cambio) | **Sin efecto medible** en el universo de 55 acciones ya filtradas — el ajuste de Prophet casi siempre pasa 0.68-0.85 |
| Screening | resto (sigma/growth/ADR) | sin cambio | No se exploraron a fondo, quedaron en el default de la app |
| Capital | `max_n` (posiciones simultáneas) | **3** | Menos operaciones = menos comisiones; 3 slots superó a 5 en la simulación de portafolio |
| Capital | `fee` | **€1/transacción** | Usado en toda la simulación de portafolio — las comisiones llegaron a comerse toda la ganancia bruta en varios escenarios |
| Selección de acciones | 55 tickers (`selected_tickers.json`) | ver backtest 7-8 | Se excluyeron 34 de las 89 del Nasdaq-100 por P&L promedio negativo en el backtest de estrategia |

**Resultado de la simulación de portafolio con esta config exacta (1 año, €1,000 iniciales, 3 slots): +19.0%** (ver backtest 14).

---

## Los backtests, en orden

### 1. `01_window_backtest.py` + `02_.../03_...analyze` — ¿Qué ventana de entrenamiento predice mejor?
Walk-forward out-of-sample: para cada ticker y fecha de corte, entrena Prophet
**solo** con datos hasta el corte (nunca ve el futuro), proyecta ~33 días, y
compara yhat contra el precio real observado después — en 6 puntos (+5 a +30
días). Se probaron 5 ventanas: 6 meses, 1, 2, 3, 4 años.

- `02_...analyze_37tickers.py`: primera corrida, 37 tickers (selección
  original de la app).
- `03_...analyze_89tickers.py`: corrida expandida a las 89 acciones del
  Nasdaq-100 completo, más un desglose de "¿subió 5%+, bajó más de 2%, o
  quedó en el medio?" para las señales de compra.

**Hallazgo clave**: 6 meses es peligroso — con tan poca historia, la
estacionalidad anual de Prophet (`yearly_seasonality=True`) no tiene
suficiente información y puede extrapolar de forma salvaje (confirmado con
AAPL: yhat cayó a $94 cuando el precio real rondaba los $330). **1 año ganó**
con el menor error en cada punto de corte y la menor tasa de fallos
catastróficos.

### 2. `04_hp_search_reduced_sample.py` → `05_hp_validate_full_scale.py` → `06_hp_search_full_finegrained.py` — Hiperparámetros de Prophet
Con la ventana ya fija en 1 año, grid search sobre `changepoint_prior_scale`
y `seasonality_mode`:
1. Búsqueda en muestra reducida (40 tickers, 6 cortes) para encontrar un
   candidato ganador rápido.
2. Validación de ese candidato contra el universo completo (89 tickers, 10
   cortes) — la mejora se redujo bastante frente a la muestra chica
   (sobreajuste a la muestra de búsqueda), lección importante.
3. Búsqueda fina final, ya a escala completa, con un rango más amplio de
   valores de `changepoint_prior_scale` y `seasonality_mode`.

**Hallazgo clave**: `changepoint_prior_scale=0.03` gana en las 4 métricas
frente al 0.05 original. `seasonality_mode="additive"` parecía mejor en
mediana de error, pero tiene una cola de riesgo grave (media de error de
hasta 37% en un combo) — se descartó a pesar de la mediana más baja.

### 3. `07_strategy_backtest.py` + `08_...analyze` — Backtest de la estrategia completa (no solo precisión de yhat)
Simula el ciclo completo: entrada cuando el precio cae bajo
`yhat × (1 − entry_dev)`, salida por Stop Loss / Take Profit / vencimiento —
en vez de solo medir si yhat acertó el número. Aquí se usaron las reglas
originales (SL=8%→3%, TP sobre yhat) antes de optimizarlas.

**Uso principal**: identificar qué acciones tienen P&L promedio negativo con
esta estrategia — resultó en la exclusión de 34 de 89 tickers (incluyendo
AAPL y MSFT, sorprendentemente) de `selected_tickers.json`.

### 4. `09_rules_search.py` + `10_...analyze` → `11_rules_search_expanded.py` + `12_...analyze` — Grid search de reglas de entrada/salida
Grid sobre `entry_dev`, `sl_pct`, `tp_pct`, `max_hold` (144 combos), con
Prophet ya fijo (se reutiliza el mismo ajuste para todas las combinaciones,
sin re-entrenar). Primero con rango estrecho de SL/TP y pocos cortes de
tiempo; luego expandido a rango más ancho de SL/TP (hasta 25%) y 25 cortes
walk-forward (~2.7 años de historia).

**Hallazgo clave**: el SL de 3% (que parecía buena idea por instinto) era en
realidad **peor** que un SL más ancho — con SL muy ajustado, el ruido normal
del precio saca de la operación antes de que la reversión funcione. La
métrica objetivo aquí fue el **profit factor** (ganancia acumulada / pérdida
acumulada), no el promedio por operación, porque el usuario prioriza pocas
operaciones (costo de transacción) sobre maximizar el P&L promedio.
`sl_pct=15%, tp_pct=20%` (Óptimo) y `sl_pct=10%, tp_pct=20%` (Conservador)
salieron como los mejores puntos, con `max_hold=15` claramente superior a 20
o 30 días.

**Nota importante de metodología**: la primera vez que se corrió el rango
expandido, los cortes de tiempo quedaron desalineados con la corrida anterior
(un bug de cálculo de fechas), haciendo que los resultados no fueran
comparables. Se corrigió fijando un ancla de fechas constante — ver los
comentarios en `11_rules_search_expanded.py`.

### 5. `13_portfolio_simulation.py` — Simulación de portafolio con capital limitado y reinversión
Ya no evalúa operaciones de forma independiente — simula el **paso real del
tiempo**, día por día, con capital limitado dividido en N "slots" (3 o 5),
reinversión del dinero de salida en la siguiente señal, y comisión de €1 por
transacción. El ajuste de Prophet se recalcula justo cuando un slot necesita
buscar una nueva acción (no en un calendario fijo), con una caché compartida
entre escenarios para no repetir ajustes innecesarios.

**Bug encontrado y corregido**: la primera versión de la caché permitía que
un ajuste hecho con datos "del futuro" (de un escenario que ya había
avanzado más en el calendario) se colara para un día anterior de otro
escenario — data leakage clásico, que producía retornos imposibles
(+1000% en un año). Se corrigió exigiendo `fit_date <= today` en la
condición de validez de la caché. **Ver los comentarios en el código para el
detalle completo** — es la lección más importante de todo este ejercicio.

### 6. `14_portfolio_simulation_screening.py` — Igual, pero con screening completo + grid de `max_dev`
Se agregaron los 5 filtros de Screening de la app real (R², sigma,
crecimiento min/max, ADR) — antes la simulación de portafolio no los
aplicaba, tratando las 55 acciones como candidatas siempre. Se agregó
también `max_dev` (cap de caída máxima) que tampoco estaba implementado.
Grid de 48 combos: `max_dev_pct` × `min_r2` × {Óptimo, Conservador} ×
{3, 5 slots}.

**Segundo bug de eficiencia (no de corrección) encontrado**: la caché
original solo guardaba el último ajuste por ticker, así que cada uno de los
48 combos re-entrenaba desde cero en vez de compartir ajustes entre sí (ya
que cada combo recorre las mismas fechas). Se corrigió guardando varios
ajustes por ticker (indexados por fecha), reduciendo de ~27,000 ajustes
estimados a **1,029 ajustes reales** para los 48 combos completos.

**Resultado ganador**: `max_dev=15%, Óptimo (SL=15%/TP=20%), 3 slots` →
**+19.0%** en el año, 141 transacciones, 50.7% win rate. `min_r2` no mostró
ningún efecto (resultados bit-idénticos entre 0.60/0.68/0.78) — el ajuste de
Prophet en este universo de 55 acciones ya preseleccionadas casi siempre
supera 0.78, así que el filtro nunca llega a excluir nada en la práctica.

---

## Lecciones aprendidas (para tener en cuenta en futuros backtests)

1. **Una muestra chica de backtest puede sobreajustarse.** El grid search de
   hiperparámetros con 40 tickers/6 cortes mostró una mejora de 13% que se
   redujo a ~3% al validar con el universo completo — siempre validar un
   "ganador" de una búsqueda reducida contra una muestra más grande antes de
   confiar en él.
2. **Cuidado con las fechas de corte al comparar corridas.** Si el ancla de
   fechas depende de un parámetro que cambia entre corridas (ej.
   `max_hold`), los cortes se desalinean y los resultados dejan de ser
   comparables aunque todo lo demás sea igual.
3. **Cuidado con cachear ajustes de modelos en simulaciones que avanzan en
   el tiempo.** Un ajuste de Prophet incluye tanto el período de
   entrenamiento como la proyección futura — reusar un ajuste sin verificar
   que su fecha de entrenamiento sea anterior al día que se está simulando
   es data leakage. Los retornos "demasiado buenos para ser verdad" son la
   señal de alarma más clara de este tipo de bug.
4. **El profit factor y el P&L promedio pueden dar recomendaciones
   distintas.** Si el usuario prioriza pocas operaciones (costo de
   transacción) sobre maximizar la ganancia promedio, hay que optimizar la
   métrica correcta desde el principio, no asumir que "más ganancia
   promedio" es siempre mejor.
5. **Un solo camino histórico (simulación de portafolio) es mucho más
   ruidoso que cientos de operaciones independientes agregadas.** Los
   resultados de `14_portfolio_simulation_screening.py` muestran patrones no
   monótonos (ej. max_dev=12% dando peor resultado que 10% y 15%) que son
   ruido del camino específico simulado, no un efecto real del parámetro —
   para conclusiones robustas sobre un parámetro, promediar varias ventanas
   de tiempo, no confiar en un solo año.

## Cómo correr los scripts

Todos requieren `yfinance`, `pandas`, `numpy`, `prophet` (mismas dependencias
que `Prophet_App/app.py`, ver `Prophet_App/requirements.txt`). Cada script
es autocontenido: descarga sus propios datos de Yahoo Finance, corre el
backtest, e imprime progreso en tiempo real (`flush=True`) para poder
seguirlo si corre en segundo plano. Los resultados se guardan en
`backtests/results/` (se crea automáticamente al correr el primer script).

Los scripts `NN_..._analyze.py` leen el CSV de resultados del script
`NN_...` correspondiente — hay que correr el backtest antes que su análisis.

```bash
cd Prophet_App/backtests
python 01_window_backtest.py
python 03_window_backtest_analyze_89tickers.py   # lee results/results.csv
```

Todos los backtests grandes (7 en adelante) tardan entre 2 y 25 minutos
según el tamaño del grid — los scripts imprimen un estimado de tiempo
restante conforme avanzan.
