"""
Prophet Mean Reversion Trader
Uso: streamlit run app.py
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import warnings, logging, json, os, base64
import requests
from datetime import date, timedelta
import paper   # cartera de papel (Prophet_App/paper.py)

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prophet Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 300px !important; }
.stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 600; padding: 8px 16px; }
/* El propio [data-baseweb="tab-list"] no queda "sticky" (Streamlit le pone
   overflow-x para el scroll horizontal de tabs). El que si funciona es su
   div contenedor directo, que no tiene clase/testid propio, por eso se
   selecciona con :has(). */
.stTabs div:has(> [data-baseweb="tab-list"]) {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: var(--background-color, #ffffff);
    box-shadow: 0 2px 4px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] {
    background: #131c27; border: 1px solid #1e2d40;
    border-radius: 8px; padding: 10px 14px;
}
</style>
""", unsafe_allow_html=True)

TODAY      = date.today()
TODAY_STR  = TODAY.strftime("%Y-%m-%d")
TODAY_TS   = pd.Timestamp(TODAY)

# ── PERSISTENCIA ───────────────────────────────────────────────────────────────
# Streamlit Community Cloud tiene disco efimero (se borra en cada sleep/redeploy),
# asi que ahi usamos el repo de GitHub como base de datos via su API de contenidos.
# Corriendo localmente (lanzar.bat) sin token configurado, usa disco local normal.
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
GITHUB_REPO   = "JuanMEcheverri/Prophet_App"
GITHUB_BRANCH = "master"
GITHUB_SUBDIR = "Prophet_App"   # ubicacion de este script dentro del repo
GITHUB_API    = "https://api.github.com"


VERSIONES_PATH = os.path.join(os.path.dirname(APP_DIR), "VERSIONES.md")


def load_versions():
    """Texto de VERSIONES.md (raiz del repo) y numero de la version actual (primera fila en negrita)."""
    try:
        with open(VERSIONES_PATH, "r", encoding="utf-8") as f:
            txt = f.read()
    except OSError:
        return None, None
    import re
    m = re.search(r"\*\*(\d+\.\d+)\*\*", txt)
    return txt, (m.group(1) if m else None)


VERSIONES_TXT, APP_VERSION = load_versions()


def _get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


GITHUB_TOKEN      = _get_secret("GITHUB_TOKEN")
USE_GITHUB_STORAGE = bool(GITHUB_TOKEN)


def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def _gh_get_file(filename):
    """Lee un archivo del repo. Devuelve (data, sha) o (None, None) si no existe."""
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{GITHUB_SUBDIR}/{filename}"
    r = requests.get(url, headers=_gh_headers(), params={"ref": GITHUB_BRANCH}, timeout=10)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    payload = r.json()
    content = base64.b64decode(payload["content"]).decode("utf-8")
    return json.loads(content), payload["sha"]


def _gh_put_file(filename, data, sha, message):
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{GITHUB_SUBDIR}/{filename}"
    body = {
        "message": message,
        "content": base64.b64encode(
            json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        ).decode("utf-8"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        body["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=body, timeout=10)
    if r.status_code == 409 and sha:
        # otro proceso escribio primero: releer sha actual y reintentar una vez
        _, fresh_sha = _gh_get_file(filename)
        body["sha"] = fresh_sha
        r = requests.put(url, headers=_gh_headers(), json=body, timeout=10)
    r.raise_for_status()
    return r.json()["content"]["sha"]


def load_json(filename, default):
    if USE_GITHUB_STORAGE:
        try:
            data, sha = _gh_get_file(filename)
            st.session_state[f"_sha_{filename}"] = sha
            return data if data is not None else default
        except Exception as e:
            st.sidebar.warning(f"No se pudo leer {filename} de GitHub: {e}")
            return default
    local_path = os.path.join(APP_DIR, filename)
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return default
    return default


def save_json(filename, data, commit_message):
    if USE_GITHUB_STORAGE:
        try:
            sha = st.session_state.get(f"_sha_{filename}")
            new_sha = _gh_put_file(filename, data, sha, commit_message)
            st.session_state[f"_sha_{filename}"] = new_sha
        except Exception as e:
            st.sidebar.warning(f"No se pudo guardar {filename} en GitHub: {e}")
        return
    local_path = os.path.join(APP_DIR, filename)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_positions():
    return load_json("positions.json", [])


def save_positions(positions):
    save_json("positions.json", positions, "Actualizar posiciones (Prophet Trader)")


def load_selection():
    return load_json("selected_tickers.json", None)


def save_selection(tickers):
    save_json("selected_tickers.json", tickers, "Actualizar seleccion de tickers (Prophet Trader)")


# Un archivo JSON independiente por seccion del sidebar (igual que positions.json
# y selected_tickers.json), para que cada grupo de parametros se recuerde solo.
# Backtest 15 (walk-forward 2021-2026, ver backtests/results/15/report_refit5.md):
# la config antigua (3/20/15/15) perdia contra QQQ en 4 de 5 años. La combo mas
# consistente fuera de muestra: entrada 4% bajo yhat, salida al volver a yhat +7%,
# SL 7%, hasta 60 dias. Aun asi NO bate a QQQ de forma clara: usar como apoyo, no
# como sistema con capital pequeño (las comisiones se comen la ventaja).
ESTRATEGIA_DEFAULTS = {
    "entry_dev_pct": 4,
    "exit_dev_pct":  7,
    "sl_pct_pct":    7,
    "max_hold":      60,
    "max_dev_pct":   15,
}
CAPITAL_DEFAULTS = {
    "capital": 1_000,
    "max_n":   3,
    "fee":     1.00,
}
# Screening revisado con 5 años de datos (entrada 4% bajo yhat, 30 dias): R2 y sigma no filtraban
# nada; el minimo de crecimiento quitaba justo las acciones que mejor revierten; el tope de
# crecimiento y el ADR si ayudan. Perfil con mejor alpha y positivo en los 6 periodos:
SCREENING_DEFAULTS = {
    "min_r2":         0.50,   # = sin filtro (minimo del slider)
    "max_sigma_pct":  15,     # = sin filtro (maximo del slider)
    "min_growth_pct": -50,    # = sin minimo
    "max_growth_pct": 20,
    "max_adr_pct":    4,
}
PROPHET_DEFAULTS = {
    "seas_mode":       "multiplicative",
    "intv_w_pct":      95,
    "chpt_scale":      0.03,
    "fc_days":         28,
    "t_start_preset":  "Ultimo 1 año",   # todos los backtests (07-15) entrenan con ventana movil de 1 año
    "t_start_custom":  "2022-01-01",
    "t_end_inp":       "",
}

T_START_PRESETS = {
    "Ultimos 6 meses": pd.DateOffset(months=6),
    "Ultimo 1 año":    pd.DateOffset(years=1),
    "Ultimos 2 años":  pd.DateOffset(years=2),
    "Ultimos 3 años":  pd.DateOffset(years=3),
}


def load_section(filename, defaults):
    saved = load_json(filename, {}) or {}
    return {**defaults, **{k: v for k, v in saved.items() if k in defaults}}


def save_section(filename, data, commit_message):
    save_json(filename, data, commit_message)


def init_section_state(filename, defaults):
    """Precarga session_state[f'cfg_{key}'] desde el JSON de esta seccion, una sola vez."""
    marker = f"_init_{filename}"
    if marker not in st.session_state:
        for k, v in load_section(filename, defaults).items():
            st.session_state[f"cfg_{k}"] = v
        st.session_state[marker] = True
        st.session_state[f"_last_saved_{filename}"] = {
            k: st.session_state[f"cfg_{k}"] for k in defaults
        }


def save_section_if_changed(filename, defaults, commit_message):
    """Guarda esta seccion solo si algun valor cambio desde el ultimo guardado."""
    current = {k: st.session_state[f"cfg_{k}"] for k in defaults}
    if current != st.session_state.get(f"_last_saved_{filename}"):
        save_section(filename, current, commit_message)
        st.session_state[f"_last_saved_{filename}"] = current

# Nasdaq-100 completo (componentes actuales)
NASDAQ100 = sorted([
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ASML","AMD","TMUS","ADBE","CSCO","PEP","QCOM","INTU","AMAT",
    "TXN","ISRG","AMGN","BKNG","MU","VRTX","REGN","PANW","KLAC","GILD",
    "LRCX","ADI","SBUX","MDLZ","MELI","SNPS","CDNS","CTAS","PAYX","ROST",
    "DDOG","ABNB","PYPL","MAR","ORLY","FAST","IDXX","WDAY","PCAR","AEP",
    "FTNT","CPRT","MNST","BIIB","DLTR","TTD","DXCM","EBAY","ENPH","WBD",
    "XEL","EXC","EA","ODFL","CRWD","NXPI","ON","ZS","VRSK","GEHC","HON",
    "FSLR","CEG","GFS","PDD","NTAP","APP","SMCI","ZBRA","KHC","ILMN",
    "OKTA","BMRN","MDB","ZM","SWKS","TEAM","MAR","MRVL","ROP","ANSS",
])

DEFAULT_SELECTION = [
    "AAPL","MSFT","GOOGL","AMZN","META","COST","NFLX","AVGO",
    "TMUS","ADBE","CSCO","PEP","QCOM","INTU","ISRG","AMGN","BKNG",
    "KLAC","GILD","LRCX","ADI","CTAS","ROST","FAST","ORLY","PCAR",
    "REGN","VRTX","XEL","AEP","EBAY","EA","MNST","MDLZ","MAR","PAYX","CDNS",
]

# Universo completo. La lista de 55 del backtest 07 se descarto en el backtest 15:
# excluir tickers por su P&L pasado no distingue del azar (percentil 68 frente a
# 200 listas aleatorias) y la alpha por ticker no persiste de un semestre al siguiente.
VALIDATED_TICKERS = sorted(set(NASDAQ100) - {"EA", "WBD", "ANSS"})   # ya no cotizan en Yahoo


def restore_validated_config():
    """Restaura estrategia/capital/screening/prophet + seleccion de tickers
    a la configuracion del backtest 15 (ver backtests/results/15/report_refit5.md),
    sobreescribiendo cualquier ajuste manual hecho despues desde el sidebar."""
    for defaults, fname in [
        (ESTRATEGIA_DEFAULTS, "estrategia.json"),
        (CAPITAL_DEFAULTS,    "capital.json"),
        (SCREENING_DEFAULTS,  "screening.json"),
        (PROPHET_DEFAULTS,    "prophet.json"),
    ]:
        for k, v in defaults.items():
            st.session_state[f"cfg_{k}"] = v
        save_section(fname, defaults, "Restaurar config backtest 15 (Prophet Trader)")
        st.session_state[f"_last_saved_{fname}"] = dict(defaults)

    st.session_state.ticker_multiselect = list(VALIDATED_TICKERS)
    save_selection(VALIDATED_TICKERS)
    st.session_state._last_saved_tickers = list(VALIDATED_TICKERS)
    st.session_state._just_restored_config = True

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Prophet Trader")
    st.caption(f"Sesion: {TODAY_STR}" + (f"  ·  v{APP_VERSION}" if APP_VERSION else ""))
    st.caption("💾 Persistencia: " + ("GitHub (cloud) ✅" if USE_GITHUB_STORAGE else "Disco local"))

    if st.session_state.pop("_just_restored_config", False):
        st.success("Configuracion del backtest 15 restaurada ✅")

    st.button(
        "↩️ Restaurar config backtest 15",
        use_container_width=True,
        on_click=restore_validated_config,
        help="Sobreescribe Estrategia/Capital/Screening/Prophet y la seleccion de "
             "acciones con la combinacion mas consistente fuera de muestra del backtest "
             "15 (walk-forward 2021-2026, ver backtests/results/15/report_refit5.md): "
             "entrada 4% bajo yhat / salida al volver a yhat +7% / SL 7% / hasta 60 dias "
             "/ cap de caida 15%, screening solo ADR<=4% y crecimiento<=20%/yr (sin minimo; "
             "R2 y sigma desactivados), changepoint_prior_scale=0.03, IC 95%, 28 dias de proyeccion, "
             "ventana movil de 1 año y el universo completo (sin lista de 55).\n\n"
             "⚠️ Ni esta config ni la anterior baten a QQQ de forma clara en cartera: "
             "con 1.000 EUR las comisiones (~13%/año) se comen la ventaja. Usar como "
             "apoyo para comprar caidas, no como sistema automatico.",
    )
    st.divider()

    init_section_state("estrategia.json", ESTRATEGIA_DEFAULTS)
    with st.expander("🎯 Estrategia", expanded=True):
        entry_dev = st.slider("Entrada: caida bajo yhat (%)", 1, 12, key="cfg_entry_dev_pct",
                              help="Comprar cuando precio < yhat × (1 − X%). yhat es el valor "
                                   "'justo' que predice Prophet para hoy — cuanto mas bajo pongas "
                                   "esto, menos señales de compra veras, pero mas seguras.\n\n"
                                   "📊 Backtest 15: caidas del 1-3% son ruido (alpha negativa en "
                                   "todos los años); a partir del 4-7% aparece algo de señal. "
                                   "Default: 4%.") / 100
        exit_dev  = st.slider("Salida: subida sobre yhat (%)", 3, 30, key="cfg_exit_dev_pct",
                              help="Vender cuando precio > yhat × (1 + X%): el precio ha vuelto "
                                   "a (y superado) el valor justo. Anclado a yhat, no a tu precio "
                                   "de entrada.\n\n📊 Backtest 15: la salida anclada a yhat +7% "
                                   "fue la mejor familia de salidas; el TP +20% sobre la entrada "
                                   "casi nunca se alcanzaba. Default: 7%.") / 100
        sl_pct    = st.slider("Stop Loss (%)", 3, 20, key="cfg_sl_pct_pct",
                              help="Si el precio cae este % por debajo de TU precio de entrada "
                                   "(no de yhat), se marca como stop loss: la tesis de reversion "
                                   "a la media fallo y toca cortar la perdida.\n\n📊 Backtest 15: "
                                   "el nivel del SL apenas cambia el resultado (7-15% equivalentes); "
                                   "un SL muy ajustado (3-6%) solo añade rotacion. Default: 7%.") / 100
        max_hold  = st.slider("Dias max en posicion", 5, 90, key="cfg_max_hold",
                              help="Dias naturales maximos con la posicion abierta. Si no llega "
                                   "a la salida por yhat ni al Stop Loss antes, se marca 'VENCER "
                                   "PRONTO' para que decidas cerrarla.\n\n📊 Backtest 15: dar "
                                   "tiempo a la reversion (30-60 dias) dobla la alpha por operacion "
                                   "frente a 10-15 dias y reduce comisiones. Default: 60.")
        max_dev   = st.slider("Cap caida maxima (%)", 5, 30, key="cfg_max_dev_pct",
                              help="No entrar si precio < yhat × (1 − cap%). Una caida mayor a "
                                   "esto suele indicar una mala noticia real (no ruido) o que el "
                                   "ajuste de Prophet ya no es confiable, asi que la accion se "
                                   "marca '⚠️ MUY BAJO' en vez de '🟢 COMPRAR'.") / 100
    save_section_if_changed("estrategia.json", ESTRATEGIA_DEFAULTS,
                            "Actualizar estrategia (Prophet Trader)")

    init_section_state("capital.json", CAPITAL_DEFAULTS)
    with st.expander("💰 Capital"):
        capital   = st.number_input("Capital total (EUR)", 200, 200_000, step=200,
                                    key="cfg_capital",
                                    help="Capital total disponible para repartir entre todas tus "
                                         "posiciones abiertas simultaneamente.")
        max_n     = st.slider("Posiciones simultaneas max", 1, 20, key="cfg_max_n",
                              help="Cuantas posiciones puedes tener abiertas al mismo tiempo. El "
                                   "capital total se divide en partes iguales (slots) entre ellas.")
        fee       = st.number_input("Comision por transaccion (EUR)", 0.0, 10.0, step=0.10,
                                    key="cfg_fee",
                                    help="Comision que cobra tu broker por cada operacion (compra "
                                         "o venta), solo informativa para estimar el costo real.")
        slot_cap  = capital / max_n
        st.caption(f"Capital por slot: €{slot_cap:,.0f}")
    save_section_if_changed("capital.json", CAPITAL_DEFAULTS,
                            "Actualizar capital (Prophet Trader)")

    init_section_state("screening.json", SCREENING_DEFAULTS)
    with st.expander("🔍 Screening"):
        min_r2     = st.slider("R² minimo Prophet", 0.50, 0.99, step=0.01, key="cfg_min_r2",
                               help="R² mide que tan bien el modelo de Prophet explica el precio "
                                    "historico: 1 = ajuste casi perfecto, 0 = sin relacion.\n\n"
                                    "📊 Backtest 15: con ventana de 1 año casi todas las acciones "
                                    "superan 0.68, asi que este filtro no cambia nada. Default: 0.50 "
                                    "(desactivado).")
        max_sigma  = st.slider("Residual sigma max (%)", 2, 15, key="cfg_max_sigma_pct",
                               help="Sigma es la desviacion estandar del error (precio real vs "
                                    "yhat) en %: mide el 'ruido' que Prophet NO logra explicar.\n\n"
                                    "📊 Backtest 15: no cambia la rentabilidad de las señales. "
                                    "Default: 15 (desactivado).") / 100
        min_growth = st.slider("Crecimiento anual min (%)", -50, 30, key="cfg_min_growth_pct",
                               help="Crecimiento anualizado minimo de la tendencia (yhat) para "
                                    "incluir la accion.\n\n📊 Backtest 15: exigir crecimiento "
                                    "positivo era contraproducente — las acciones con tendencia "
                                    "plana o bajista son justo las que mejor revierten tras una "
                                    "caida (+1.3%/op frente a un dia normal, positivo los 5 años). "
                                    "Default: -50 (sin minimo).") / 100
        max_growth = st.slider("Crecimiento anual max (%)", 5, 120, key="cfg_max_growth_pct",
                               help="Crecimiento anualizado maximo permitido.\n\n📊 Backtest 15: "
                                    "las acciones con tendencia explosiva (>35%/año) tienen alpha "
                                    "negativa tras una caida: no revierten, siguen cayendo o ya "
                                    "venian de subir demasiado. Default: 20.") / 100
        max_adr    = st.slider("ADR diario max (%)", 1, 10, key="cfg_max_adr_pct",
                               help="Average Daily Range: rango (maximo−minimo)/cierre promedio "
                                    "de los ultimos 90 dias.\n\n📊 Backtest 15: no sube la "
                                    "rentabilidad pero reduce la caida tipica de una operacion "
                                    "fallida de -8.6% a -4.7%. Default: 4.") / 100
    save_section_if_changed("screening.json", SCREENING_DEFAULTS,
                            "Actualizar screening (Prophet Trader)")

    init_section_state("prophet.json", PROPHET_DEFAULTS)
    FC_DAYS_OPTIONS = [7, 14, 21, 28, 35, 42, 49, 56]
    if st.session_state.get("cfg_fc_days") not in FC_DAYS_OPTIONS:
        # migracion: valores guardados antes de este cambio (ej. 30, 60, 180)
        # se ajustan al multiplo de semana mas cercano.
        prev = st.session_state.get("cfg_fc_days", 28)
        st.session_state.cfg_fc_days = min(FC_DAYS_OPTIONS, key=lambda d: abs(d - prev))
    with st.expander("🔮 Prophet"):
        seas_mode  = st.selectbox("Seasonality mode", ["multiplicative", "additive"],
                                  key="cfg_seas_mode",
                                  help="Como se combina la estacionalidad con la tendencia. "
                                       "'multiplicative': el efecto estacional crece o decrece "
                                       "proporcional al precio (recomendado para acciones, que "
                                       "crecen exponencialmente). 'additive': el efecto estacional "
                                       "es un monto fijo, independiente del nivel de precio.\n\n"
                                       "📊 Grid search (89 acciones, 890 backtests por combo): "
                                       "'additive' da mediana de error ~3-4% mejor, pero con cola de "
                                       "riesgo grave (media de error hasta 37% en el peor caso — "
                                       "fallos raros pero catastroficos). 'multiplicative' se "
                                       "mantiene estable en todos los casos probados. "
                                       "**Recomendado y default: multiplicative.**")
        intv_w     = st.slider("Intervalo de confianza (%)", 80, 99, key="cfg_intv_w_pct",
                               help="Ancho de la banda de incertidumbre (sombreada en los charts) "
                                    "alrededor de yhat. Prophet la calcula simulando que los "
                                    "cambios de tendencia futuros se parecen a los del pasado. "
                                    "Solo afecta la visualizacion, no las señales de compra/venta.") / 100
        chpt_scale = st.slider("Flexibilidad de tendencia", 0.01, 0.50, step=0.01,
                               key="cfg_chpt_scale",
                               help="changepoint_prior_scale — controla cuanto puede doblarse la "
                                    "linea de tendencia (yhat) en los 'changepoints' que Prophet "
                                    "detecta. Mas alto = tendencia mas flexible/pegada al precio "
                                    "(riesgo de sobreajuste); mas bajo = tendencia mas rigida "
                                    "(riesgo de no captar cambios reales de comportamiento).\n\n"
                                    "📊 Grid search (89 acciones, 890 backtests por valor, con "
                                    "seasonality_mode=multiplicative): 0.03 gano en mediana, media "
                                    "Y tasa de outliers frente al 0.05 original, y frente a 0.07/"
                                    "0.10/0.15 probados. **Recomendado y default: 0.03.**")
        fc_days    = st.radio(
            "Dias de proyeccion futura",
            FC_DAYS_OPTIONS,
            format_func=lambda d: f"{d // 7} semana" + ("s" if d // 7 > 1 else ""),
            horizontal=True,
            key="cfg_fc_days",
            help="Cuantos dias hacia adelante proyecta Prophet el yhat futuro (maximo 60 dias, "
                 "en pasos de 1 semana). Define el horizonte de las señales y de la pestaña "
                 "Proyecciones.")
        t_start_preset = st.selectbox(
            "Inicio entrenamiento",
            list(T_START_PRESETS.keys()) + ["Personalizado"],
            key="cfg_t_start_preset",
            help="Desde cuando se toman datos historicos para entrenar (fit) el modelo Prophet "
                 "de cada accion. Elige un rango rapido o 'Personalizado' para escribir una "
                 "fecha exacta.\n\n"
                 "📊 Backtest walk-forward (89 acciones, 890 trades simulados por ventana): "
                 "'Ultimo 1 año' fue el mejor tanto en precision de yhat (menor error en "
                 "cada punto de corte, menos fallos catastroficos) como en P&L real de la "
                 "estrategia completa (mayor win-rate y expectativa por operacion). "
                 "'Ultimos 6 meses' es riesgoso: la estacionalidad anual no tiene suficiente "
                 "historia y puede extrapolar mal. **Recomendado: Ultimo 1 año.**",
        )
        if t_start_preset == "Personalizado":
            t_start = st.text_input("Fecha exacta (AAAA-MM-DD)", key="cfg_t_start_custom",
                                    help="Formato AAAA-MM-DD, ej. 2022-01-01.")
        else:
            t_start = (TODAY_TS - T_START_PRESETS[t_start_preset]).strftime("%Y-%m-%d")
            st.caption(f"Entrenando desde {t_start}")
        t_end_inp  = st.text_input("Fin entrenamiento (vacio = hoy)", key="cfg_t_end_inp",
                                   help="Fecha hasta la cual se usan datos para entrenar el "
                                        "modelo. Dejalo vacio para usar todos los datos hasta hoy. "
                                        "Poner una fecha pasada sirve para backtesting: entrenas "
                                        "con datos viejos y comparas el yhat contra lo que "
                                        "realmente paso despues.")
    save_section_if_changed("prophet.json", PROPHET_DEFAULTS,
                            "Actualizar config Prophet (Prophet Trader)")

    with st.expander("📋 Acciones Nasdaq-100"):
        if "ticker_multiselect" not in st.session_state:
            saved_selection = load_selection()
            st.session_state.ticker_multiselect = (
                saved_selection if saved_selection is not None else DEFAULT_SELECTION
            )
            st.session_state._last_saved_tickers = list(st.session_state.ticker_multiselect)

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            if st.button("Todas", use_container_width=True, key="btn_all"):
                st.session_state.ticker_multiselect = NASDAQ100
        with col_sel2:
            if st.button("Ninguna", use_container_width=True, key="btn_none"):
                st.session_state.ticker_multiselect = []

        selected_tickers = st.multiselect(
            "Selecciona acciones",
            options=NASDAQ100,
            key="ticker_multiselect",
            label_visibility="collapsed",
            help="Tu seleccion se guarda automaticamente y se recupera aunque cierres el "
                 "navegador o reinicies la app.",
        )
        # Solo guardar si de verdad cambio — evita un commit a GitHub en cada rerun
        if selected_tickers != st.session_state.get("_last_saved_tickers"):
            save_selection(selected_tickers)
            st.session_state._last_saved_tickers = list(selected_tickers)
        tickers = tuple(sorted(set(selected_tickers)))
        st.caption(f"{len(tickers)} acciones seleccionadas — guardadas permanentemente ✅")

    st.divider()
    train_btn  = st.button("Entrenar Prophet", type="primary", use_container_width=True)
    refresh_px = st.button("Actualizar precios", use_container_width=True)

t_end_str = t_end_inp.strip() if t_end_inp.strip() else TODAY_STR

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "cache_buster" not in st.session_state:
    st.session_state.cache_buster = 0
if "positions" not in st.session_state:
    st.session_state.positions = load_positions()
if "trained" not in st.session_state:
    st.session_state.trained = False

if train_btn:
    st.session_state.cache_buster += 1
    st.session_state.trained = False

# ── CACHED: TRAINING ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(tickers, t_start, t_end, seas_mode, intv_w, chpt_scale,
                 min_r2, max_sigma, min_growth, max_growth, max_adr, fc_days,
                 _buster=0):
    all_t = list(tickers)
    end_dl = (pd.Timestamp(t_end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    raw = yf.download(all_t, start=t_start, end=end_dl,
                      auto_adjust=True, progress=False)

    # Normalize columns for single vs multi ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        high  = raw["High"]
        low   = raw["Low"]
    else:
        close = raw[["Close"]].rename(columns={"Close": all_t[0]})
        high  = raw[["High"]].rename(columns={"High": all_t[0]})
        low   = raw[["Low"]].rename(columns={"Low": all_t[0]})

    def strip(s):
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s

    sdata = {}
    log   = []

    for t in sorted(all_t):
        try:
            cl = strip(close[t].dropna())
            hi = strip(high[t].dropna())
            lo = strip(low[t].dropna())

            # con la ventana de 1 año llegan ~251 cierres; con festivos o un dia sin dato
            # el umbral 250 descartaba tickers validos. Mismo minimo que el backtest 15.
            if len(cl) < 200:
                log.append({"Ticker": t, "Pass": False, "Motivo": "Datos insuficientes (<200)"})
                continue

            # ADR (ultimos 90 dias)
            tail = min(90, len(cl))
            adr  = float(((hi.tail(tail) - lo.tail(tail)) /
                          cl.tail(tail).clip(lower=0.01)).mean())
            if adr > max_adr:
                log.append({"Ticker": t, "Pass": False,
                            "R2": None, "Sigma%": None, "Growth%/yr": None,
                            "ADR%": round(adr * 100, 1),
                            "Motivo": f"ADR {adr*100:.1f}% > {max_adr*100:.0f}%"})
                continue

            df_fit = pd.DataFrame({"ds": cl.index.normalize(), "y": cl.values}).dropna()

            m = Prophet(
                interval_width=intv_w,
                seasonality_mode=seas_mode,
                growth="linear",
                changepoint_prior_scale=chpt_scale,
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=False,
            )
            m.fit(df_fit)

            future = m.make_future_dataframe(periods=fc_days + 15)
            pred   = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            pred["ds"] = pred["ds"].dt.normalize()

            fc_dict = {
                row.ds: (row.yhat, row.yhat_lower, row.yhat_upper)
                for row in pred.itertuples(index=False)
            }

            # Quality metrics
            df_idx = df_fit.set_index("ds")
            common = [d for d in df_idx.index if d in fc_dict]
            y_is   = np.array([df_idx.loc[d, "y"] for d in common])
            yh_is  = np.array([fc_dict[d][0] for d in common])
            resids = (y_is - yh_is) / np.clip(yh_is, 0.01, None)
            r2     = float(np.corrcoef(y_is, yh_is)[0, 1] ** 2)
            sigma  = float(np.std(resids))

            fc_list = sorted(fc_dict.items())
            y0   = fc_list[0][1][0];  y1 = fc_list[-1][1][0]
            nyrs = (fc_list[-1][0] - fc_list[0][0]).days / 365.25
            growth = (y1 / y0) ** (1 / nyrs) - 1 if nyrs > 0 and y0 > 0 else 0

            passes = (r2 >= min_r2 and sigma <= max_sigma
                      and min_growth <= growth <= max_growth)

            motivo = "OK"
            if not passes:
                if r2 < min_r2:           motivo = f"R2 {r2:.2f} < {min_r2}"
                elif sigma > max_sigma:   motivo = f"Sigma {sigma*100:.1f}% > {max_sigma*100:.0f}%"
                else:                     motivo = f"Growth {growth*100:.0f}%/yr fuera rango"

            log.append({
                "Ticker": t, "Pass": passes,
                "R2": round(r2, 3),
                "Sigma%": round(sigma * 100, 1),
                "Growth%/yr": round(growth * 100, 1),
                "ADR%": round(adr * 100, 1),
                "Motivo": motivo,
            })

            if passes:
                sdata[t] = {
                    "fc_dict": fc_dict,
                    "df_hist": df_fit,
                    "r2": r2, "sigma": sigma,
                    "growth": growth, "adr": adr,
                }
        except Exception as e:
            log.append({"Ticker": t, "Pass": False, "Motivo": str(e)[:100]})

    return sdata, pd.DataFrame(log)


@st.cache_data(ttl=900, show_spinner=False)
def get_prices(tickers, _buster=0):
    """Precios actuales — TTL 15 min. Devuelve ultimo cierre, apertura de ese dia y cierre anterior."""
    raw = yf.download(list(tickers), period="5d", auto_adjust=True, progress=False)
    out = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                cl = raw["Close"][t].dropna(); op = raw["Open"][t]
            else:
                cl = raw["Close"].dropna(); op = raw["Open"]
            if cl.index.tz is not None:
                cl.index = cl.index.tz_localize(None); op.index = op.index.tz_localize(None)
            if len(cl):
                o = op.get(cl.index[-1])
                out[t] = {"px": float(cl.iloc[-1]), "date": cl.index[-1].date(),
                          "open": (None if o is None or pd.isna(o) else float(o)),
                          "prev_px": (float(cl.iloc[-2]) if len(cl) > 1 else None)}
        except:
            pass
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def get_fx_rate(d):
    """EUR/USD (dolares por euro) al cierre de la fecha d o del ultimo dia anterior con dato."""
    try:
        ts = pd.Timestamp(d)
        raw = yf.download("EURUSD=X", start=(ts - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                          end=(ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
        cl = raw["Close"].squeeze().dropna()
        if cl.index.tz is not None:
            cl.index = cl.index.tz_localize(None)
        cl = cl[cl.index <= ts]
        return float(cl.iloc[-1]) if len(cl) else None
    except Exception:
        return None


# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_yhat_at(fc_dict, day):
    dn = pd.Timestamp(day).normalize()
    if dn in fc_dict:
        return fc_dict[dn]
    keys = sorted(k for k in fc_dict if k <= dn)
    return fc_dict[keys[-1]] if keys else None


def classify(dev_pct, entry_pct, exit_pct, max_pct):
    if dev_pct <= -max_pct * 100:   return "⚠️ MUY BAJO"
    if dev_pct <= -entry_pct * 100: return "🟢 COMPRAR"
    if dev_pct >= exit_pct * 100:   return "🔴 SOBRE COMPRADO"
    return "⚪ NEUTRAL"


def build_chart(t, sd, cur_px, t_end_str, entry_dev, exit_dev, fc_days):
    df_hist = sd["df_hist"]
    fc_dict = sd["fc_dict"]
    t_end_ts = pd.Timestamp(t_end_str)
    today_ts = pd.Timestamp(TODAY)

    fc_sorted = sorted(fc_dict.items())
    fc_hist   = [(d, v) for d, v in fc_sorted if d <= t_end_ts]
    fc_fut    = [(d, v) for d, v in fc_sorted if d > t_end_ts]

    fig = go.Figure()

    # Training zone shading
    if df_hist["ds"].min() < t_end_ts:
        fig.add_vrect(
            x0=df_hist["ds"].min().strftime("%Y-%m-%d"),
            x1=t_end_ts.strftime("%Y-%m-%d"),
            fillcolor="rgba(74,96,120,0.07)", line_width=0,
        )

    # CI band — historical
    if fc_hist:
        dh = [d for d, _ in fc_hist]
        fig.add_trace(go.Scatter(
            x=dh + dh[::-1],
            y=[v[2] for _, v in fc_hist] + [v[1] for _, v in reversed(fc_hist)],
            fill="toself", fillcolor="rgba(79,172,222,0.08)",
            line_width=0, showlegend=False, hoverinfo="skip",
        ))

    # CI band — future
    if fc_fut:
        df = [d for d, _ in fc_fut]
        yhat_f = [v[0] for _, v in fc_fut]
        fig.add_trace(go.Scatter(
            x=df + df[::-1],
            y=[v[2] for _, v in fc_fut] + [v[1] for _, v in reversed(fc_fut)],
            fill="toself", fillcolor="rgba(79,172,222,0.13)",
            line_width=0, name="IC 95%", hoverinfo="skip",
        ))
        # Entry/exit threshold bands
        fig.add_trace(go.Scatter(
            x=df, y=[y * (1 - entry_dev) for y in yhat_f],
            line=dict(color="#27ae60", dash="dash", width=1.2),
            name=f"Umbral compra −{entry_dev*100:.0f}%", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df, y=[y * (1 + exit_dev) for y in yhat_f],
            line=dict(color="#e74c3c", dash="dash", width=1.2),
            name=f"Umbral venta +{exit_dev*100:.0f}%", hoverinfo="skip",
        ))
        # yhat future
        fig.add_trace(go.Scatter(
            x=df, y=yhat_f,
            line=dict(color="#4facde", width=2),
            name="Prophet yhat (proyeccion)",
        ))

    # yhat historical
    if fc_hist:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in fc_hist], y=[v[0] for _, v in fc_hist],
            line=dict(color="#4facde", width=1.5, dash="dot"),
            name="Prophet yhat (ajustado)",
        ))

    # Historical price
    fig.add_trace(go.Scatter(
        x=df_hist["ds"], y=df_hist["y"],
        line=dict(color="#4b6584", width=1.5),
        name="Precio historico",
    ))

    # Cutoff line — pasar como string evita el bug de aritmética de Timestamps en Plotly 6
    if t_end_ts > df_hist["ds"].min():
        cutoff_str = t_end_ts.strftime("%Y-%m-%d")
        fig.add_vline(x=cutoff_str, line_dash="dash",
                      line_color="rgba(79,172,222,0.4)")
        fig.add_annotation(
            x=cutoff_str, y=1, yref="paper",
            text="Fin entrenamiento", showarrow=False,
            xanchor="right", yanchor="top",
            font=dict(size=10, color="#4facde"),
            bgcolor="rgba(0,0,0,0)",
        )

    # Today's price marker
    if cur_px is not None:
        yht = get_yhat_at(fc_dict, today_ts)
        dev = (cur_px / yht[0] - 1) * 100 if yht else 0
        color = ("#27ae60" if dev <= -entry_dev * 100 else
                 "#e74c3c" if dev >= exit_dev * 100 else "#f0a500")
        fig.add_trace(go.Scatter(
            x=[today_ts], y=[cur_px],
            mode="markers+text",
            marker=dict(size=11, color=color, symbol="circle",
                        line=dict(width=2, color="white")),
            text=[f"Hoy ${cur_px:.2f}<br>({dev:+.1f}%)"],
            textposition="top center",
            textfont=dict(size=10, color=color),
            name=f"Precio hoy: ${cur_px:.2f} ({dev:+.1f}% yhat)",
        ))

    fig.update_layout(
        title=dict(
            text=(f"<b>{t}</b> — R²={sd['r2']:.2f}  "
                  f"σ={sd['sigma']*100:.1f}%  "
                  f"g={sd['growth']*100:.1f}%/yr  "
                  f"ADR={sd['adr']*100:.1f}%"),
            font_size=13,
        ),
        height=500,
        paper_bgcolor="#0c1118",
        plot_bgcolor="#131c27",
        font_color="#c8d6e5",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font_size=10, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1e2d40", showgrid=True, showspikes=True),
        yaxis=dict(gridcolor="#1e2d40", showgrid=True, tickprefix="$"),
        margin=dict(l=50, r=20, t=70, b=40),
        hovermode="x unified",
    )
    return fig


# ── TRAINING ─────────────────────────────────────────────────────────────────
with st.spinner("Entrenando Prophet — la primera vez tarda 1-2 min..."):
    sdata, screen_log = train_models(
        tickers, t_start, t_end_str, seas_mode, intv_w, chpt_scale,
        min_r2, max_sigma, min_growth, max_growth, max_adr, fc_days,
        _buster=st.session_state.cache_buster,
    )

if not sdata:
    st.error("Ninguna accion paso el screening con los parametros actuales. Ajusta los criterios en la barra lateral.")
    st.stop()

px_buster = st.session_state.cache_buster + (1 if refresh_px else 0)
cur_prices = get_prices(tuple(sorted(sdata.keys())), _buster=px_buster)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_hoy, tab_screen, tab_chart, tab_proj, tab_pos, tab_paper1, tab_paper2, tab_docs, tab_ver = st.tabs([
    "🚨 Señales de Hoy",
    "🔍 Screening",
    "📊 Charts",
    "📋 Proyecciones",
    "💼 Mis Posiciones",
    "🧪 Claude · Backtest 15",
    "🧪 Claude · Combinado",
    "📚 Como funciona Prophet",
    "ℹ️ Versiones",
])

with tab_ver:
    st.subheader("Versiones de la app" + (f" · actual v{APP_VERSION}" if APP_VERSION else ""))
    if VERSIONES_TXT:
        st.markdown(VERSIONES_TXT.split("\n", 1)[1] if VERSIONES_TXT.startswith("#") else VERSIONES_TXT)
    else:
        st.info("No se encontro VERSIONES.md en la raiz del repositorio.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEÑALES DE HOY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hoy:
    st.subheader(f"Señales para el {TODAY_STR}")
    st.caption(
        f"Comprar cuando precio < yhat×{1-entry_dev:.2f}  |  "
        f"Vender cuando > yhat×{1+exit_dev:.2f}  |  "
        f"SL −{sl_pct*100:.0f}%  |  MaxHold {max_hold}d"
    )

    rows = []
    for t, sd in sorted(sdata.items()):
        info = cur_prices.get(t, {})
        px   = info.get("px")
        yht  = get_yhat_at(sd["fc_dict"], TODAY_TS)
        if yht is None or px is None:
            continue
        yhat_v, yhat_lo, yhat_hi = yht
        dev     = (px / yhat_v - 1) * 100
        thr_buy = yhat_v * (1 - entry_dev)
        thr_sel = yhat_v * (1 + exit_dev)
        sl_px   = px * (1 - sl_pct)
        dist_tp = (thr_sel / px - 1) * 100
        dist_sl = (sl_px / px - 1) * 100
        signal  = classify(dev, entry_dev, exit_dev, max_dev)
        rows.append({
            "Ticker":       t,
            "Precio Hoy":   px,
            "yhat":         round(yhat_v, 2),
            "Dev%":         round(dev, 2),
            "Señal":        signal,
            "Thr Compra":   round(thr_buy, 2),
            "Thr Venta":    round(thr_sel, 2),
            "SL Ref":       round(sl_px, 2),
            "Dist TP%":     round(dist_tp, 1),
            "Dist SL%":     round(dist_sl, 1),
            "R2":           round(sd["r2"], 2),
        })

    df_hoy = pd.DataFrame(rows).sort_values("Dev%")

    # KPI metrics
    buy_zone = df_hoy[df_hoy["Señal"] == "🟢 COMPRAR"]
    sob_zone = df_hoy[df_hoy["Señal"] == "🔴 SOBRE COMPRADO"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acciones analizadas", len(df_hoy))
    col2.metric("En zona de compra 🟢", len(buy_zone))
    col3.metric("Sobre compradas 🔴", len(sob_zone))
    col4.metric("Neutras ⚪", len(df_hoy) - len(buy_zone) - len(sob_zone))

    # Highlight buy zone first
    if not buy_zone.empty:
        st.success(f"**{len(buy_zone)} acciones en zona de compra:**  " +
                   "  |  ".join(f"**{r['Ticker']}** {r['Dev%']:+.1f}%"
                                for _, r in buy_zone.iterrows()))

    st.dataframe(
        df_hoy,
        use_container_width=True,
        height=420,
        hide_index=True,
        column_config={
            "Ticker":     st.column_config.TextColumn("Ticker", width=70),
            "Precio Hoy": st.column_config.NumberColumn("Precio $", format="$%.2f"),
            "yhat":       st.column_config.NumberColumn("yhat $", format="$%.2f"),
            "Dev%":       st.column_config.NumberColumn("Dev%", format="%.1f%%"),
            "Señal":      st.column_config.TextColumn("Señal", width=140),
            "Thr Compra": st.column_config.NumberColumn("Thr Compra $", format="$%.2f"),
            "Thr Venta":  st.column_config.NumberColumn("Thr Venta $", format="$%.2f"),
            "SL Ref":     st.column_config.NumberColumn("SL Ref $", format="$%.2f"),
            "Dist TP%":   st.column_config.NumberColumn("Dist TP%", format="+%.1f%%"),
            "Dist SL%":   st.column_config.NumberColumn("Dist SL%", format="%.1f%%"),
            "R2":         st.column_config.ProgressColumn("R²", min_value=0, max_value=1, format="%.2f"),
        },
    )
    st.caption(
        "Dev% = (Precio / yhat − 1) × 100  |  "
        "Thr Compra = yhat × (1 − entry_dev)  |  "
        "Dist TP% = distancia del precio actual al umbral de venta"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCREENING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_screen:
    st.subheader("Resultados de Screening")
    st.caption(
        f"Criterios: R²≥{min_r2}  |  σ≤{max_sigma*100:.0f}%  |  "
        f"Crecimiento {min_growth*100:.0f}–{max_growth*100:.0f}%/yr  |  ADR≤{max_adr*100:.0f}%"
    )

    col1, col2 = st.columns(2)
    with col1:
        show_pass = st.checkbox("Solo las que pasan", value=False)

    log_show = screen_log.copy()
    if show_pass:
        log_show = log_show[log_show["Pass"] == True]

    # Sort: passes first, then by R2
    log_show = log_show.sort_values(["Pass", "R2"], ascending=[False, False])

    st.dataframe(
        log_show,
        use_container_width=True,
        height=500,
        hide_index=True,
        column_config={
            "Ticker":       st.column_config.TextColumn("Ticker", width=70),
            "Pass":         st.column_config.CheckboxColumn("Pasa"),
            "R2":           st.column_config.ProgressColumn("R²", min_value=0, max_value=1, format="%.3f"),
            "Sigma%":       st.column_config.NumberColumn("Sigma%", format="%.1f%%"),
            "Growth%/yr":   st.column_config.NumberColumn("Crec%/yr", format="%.1f%%"),
            "ADR%":         st.column_config.NumberColumn("ADR%", format="%.1f%%"),
            "Motivo":       st.column_config.TextColumn("Motivo", width=220),
        },
    )
    n_pass = screen_log["Pass"].sum()
    n_fail = len(screen_log) - n_pass
    st.caption(f"✅ {n_pass} pasan  |  ❌ {n_fail} no pasan  |  Total: {len(screen_log)}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.subheader("Analisis Prophet por accion")

    buy_zone_tickers = sorted(df_hoy[df_hoy["Señal"] == "🟢 COMPRAR"]["Ticker"].tolist())

    view_mode = st.radio(
        "Vista",
        ["Una accion", "Todas las acciones", f"🟢 Solo zona de compra ({len(buy_zone_tickers)})"],
        horizontal=True,
    )

    if view_mode == "Una accion":
        selected_t = st.selectbox("Selecciona accion", sorted(sdata.keys()))

    def render_chart(t):
        sd = sdata[t]
        cur_px = cur_prices.get(t, {}).get("px")
        fig = build_chart(t, sd, cur_px, t_end_str, entry_dev, exit_dev, fc_days)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Mini stats below chart
        if cur_px is not None:
            yht = get_yhat_at(sd["fc_dict"], TODAY_TS)
            if yht:
                dev = (cur_px / yht[0] - 1) * 100
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Precio actual", f"${cur_px:.2f}")
                c2.metric("yhat hoy", f"${yht[0]:.2f}")
                c3.metric("Desviacion", f"{dev:+.1f}%")
                c4.metric("Umbral compra", f"${yht[0]*(1-entry_dev):.2f}")
                c5.metric("Umbral venta", f"${yht[0]*(1+exit_dev):.2f}")

    if view_mode == "Todas las acciones":
        for t in sorted(sdata.keys()):
            with st.expander(f"📈 {t}", expanded=False):
                render_chart(t)
    elif view_mode.startswith("🟢 Solo zona de compra"):
        if not buy_zone_tickers:
            st.info("Ninguna accion esta en zona de compra con los parametros actuales.")
        else:
            st.success(f"**{len(buy_zone_tickers)} acciones en zona de compra:** " +
                       ", ".join(buy_zone_tickers))
            for t in buy_zone_tickers:
                with st.expander(f"📈 {t}", expanded=True):
                    render_chart(t)
    else:
        if selected_t:
            render_chart(selected_t)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROYECCIONES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_proj:
    st.subheader("Tabla de Proyecciones Prophet")
    st.caption(
        "Muestra el yhat futuro por accion y dia. Dev% = como estaria el precio de hoy "
        "respecto al yhat de ese dia futuro."
    )

    # Filters
    fcol1, fcol2, fcol3, fcol4 = st.columns([2, 2, 1, 1])
    with fcol1:
        sel_tickers = st.multiselect(
            "Acciones", sorted(sdata.keys()), default=sorted(sdata.keys()),
            key="proj_tickers"
        )
    with fcol2:
        date_from = st.date_input("Desde", value=TODAY, key="proj_from")
        date_to   = st.date_input("Hasta", value=TODAY + timedelta(days=fc_days), key="proj_to")
    with fcol3:
        sel_signals = st.multiselect(
            "Señal", ["🟢 COMPRAR", "⚪ NEUTRAL", "🔴 SOBRE COMPRADO", "⚠️ MUY BAJO"],
            default=["🟢 COMPRAR", "⚪ NEUTRAL", "🔴 SOBRE COMPRADO", "⚠️ MUY BAJO"],
            key="proj_signals"
        )
    with fcol4:
        sort_by = st.selectbox("Ordenar por",
            ["Dev% (menor primero)", "Fecha", "Ticker", "yhat"],
            key="proj_sort"
        )

    rows_proj = []
    for t in sel_tickers:
        sd  = sdata[t]
        cpx = cur_prices.get(t, {}).get("px")
        for ds, (yhat_v, yhat_lo, yhat_hi) in sorted(sd["fc_dict"].items()):
            d = ds.date()
            if d < date_from or d > date_to:
                continue
            dev = (cpx / yhat_v - 1) * 100 if cpx and yhat_v else None
            sig = classify(dev, entry_dev, exit_dev, max_dev) if dev is not None else "—"
            if sig not in sel_signals and sel_signals:
                continue
            days_from_today = (d - TODAY).days
            rows_proj.append({
                "Ticker":        t,
                "Fecha":         d,
                "Dias":          days_from_today,
                "yhat":          round(yhat_v, 2),
                "yhat_lo":       round(yhat_lo, 2),
                "yhat_hi":       round(yhat_hi, 2),
                "Thr Compra":    round(yhat_v * (1 - entry_dev), 2),
                "Thr Venta":     round(yhat_v * (1 + exit_dev), 2),
                "Precio Hoy":    round(cpx, 2) if cpx else None,
                "Dev%":          round(dev, 2) if dev is not None else None,
                "Señal":         sig,
            })

    if rows_proj:
        df_proj = pd.DataFrame(rows_proj)
        sort_map = {
            "Dev% (menor primero)": ("Dev%", True),
            "Fecha":  ("Fecha", True),
            "Ticker": ("Ticker", True),
            "yhat":   ("yhat", True),
        }
        col_s, asc_s = sort_map[sort_by]
        df_proj = df_proj.sort_values(col_s, ascending=asc_s)

        st.dataframe(
            df_proj,
            use_container_width=True,
            height=520,
            hide_index=True,
            column_config={
                "Ticker":     st.column_config.TextColumn("Ticker", width=70),
                "Fecha":      st.column_config.DateColumn("Fecha", width=100),
                "Dias":       st.column_config.NumberColumn("Dias", format="%d"),
                "yhat":       st.column_config.NumberColumn("yhat $", format="$%.2f"),
                "yhat_lo":    st.column_config.NumberColumn("yhat Lo $", format="$%.2f"),
                "yhat_hi":    st.column_config.NumberColumn("yhat Hi $", format="$%.2f"),
                "Thr Compra": st.column_config.NumberColumn("Thr Compra $", format="$%.2f"),
                "Thr Venta":  st.column_config.NumberColumn("Thr Venta $", format="$%.2f"),
                "Precio Hoy": st.column_config.NumberColumn("Precio Hoy $", format="$%.2f"),
                "Dev%":       st.column_config.NumberColumn("Dev%", format="%.1f%%"),
                "Señal":      st.column_config.TextColumn("Señal", width=140),
            },
        )
        st.caption(f"{len(df_proj)} filas  |  {len(sel_tickers)} acciones")

        csv = df_proj.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV", csv,
                           file_name=f"proyecciones_{TODAY_STR}.csv",
                           mime="text/csv")
    else:
        st.info("No hay proyecciones con los filtros seleccionados.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — POSICIONES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pos:
    st.subheader("Seguimiento de Posiciones")
    st.caption("Precios en EUR o USD: la app convierte con el EUR/USD del dia (cierre de Yahoo) y guarda los dos. "
               "Los calculos de SL / salida se hacen en USD, que es la moneda de la accion.")

    def _price_both(px, ccy, d):
        """Devuelve (precio_usd, precio_eur, fx) a partir de un precio en la moneda indicada."""
        fx = get_fx_rate(d)
        if fx is None:
            return (px, None, None) if ccy == "USD" else (None, px, None)
        return (px, px / fx, fx) if ccy == "USD" else (px * fx, px, fx)

    open_positions   = [(i, p) for i, p in enumerate(st.session_state.positions) if not p.get("cerrada")]
    closed_positions = [(i, p) for i, p in enumerate(st.session_state.positions) if p.get("cerrada")]

    # ── Agregar posicion (entrada) ───────────────────────────────────────────
    with st.expander("➕ Registrar entrada (compra)", expanded=False):
        with st.form("add_pos", clear_on_submit=True):
            pc1, pc2, pc3, pc4, pc5 = st.columns(5)
            with pc1:
                pos_ticker = st.selectbox("Ticker", sorted(set(sdata.keys()) | {p["ticker"] for _, p in open_positions}))
            with pc2:
                pos_date   = st.date_input("Fecha entrada", value=TODAY)
            with pc3:
                pos_ccy    = st.selectbox("Moneda del precio", ["EUR", "USD"])
            with pc4:
                pos_px     = st.number_input("Precio por accion", min_value=0.01, value=100.0, step=0.01)
            with pc5:
                pos_cap    = st.number_input("Capital invertido (EUR)", min_value=1.0, value=float(slot_cap), step=10.0)
            pos_note = st.text_input("Nota (opcional)")
            submitted = st.form_submit_button("Agregar", type="primary")
            if submitted:
                usd, eur, fx = _price_both(pos_px, pos_ccy, pos_date)
                if usd is None:
                    st.error("No pude obtener el EUR/USD de esa fecha; introduce el precio en USD.")
                else:
                    st.session_state.positions.append({
                        "ticker":    pos_ticker,
                        "fecha_entrada": str(pos_date),
                        "precio_entrada": round(usd, 4),
                        "precio_entrada_eur": (round(eur, 4) if eur is not None else None),
                        "fx_entrada": fx, "moneda_entrada": pos_ccy,
                        "capital":   pos_cap,
                        "nota":      pos_note,
                    })
                    save_positions(st.session_state.positions)
                    st.success(f"Entrada {pos_ticker} guardada: {usd:.2f} $" + (f" = {eur:.2f} € (EUR/USD {fx:.4f})" if eur is not None else ""))

    # ── Registrar salida (venta) ─────────────────────────────────────────────
    with st.expander("➖ Registrar salida (venta)", expanded=False):
        if not open_positions:
            st.info("No hay posiciones abiertas que cerrar.")
        else:
            with st.form("close_pos", clear_on_submit=True):
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    sel_i = st.selectbox("Posicion", [i for i, _ in open_positions],
                                         format_func=lambda i: f"{st.session_state.positions[i]['ticker']} (entrada {st.session_state.positions[i]['fecha_entrada']})")
                with sc2:
                    exit_date = st.date_input("Fecha salida", value=TODAY)
                with sc3:
                    exit_ccy  = st.selectbox("Moneda del precio", ["EUR", "USD"], key="exit_ccy")
                with sc4:
                    exit_px   = st.number_input("Precio de venta por accion", min_value=0.01, value=100.0, step=0.01)
                exit_note = st.text_input("Motivo (take profit, stop loss, tiempo, otro)")
                if st.form_submit_button("Cerrar posicion", type="primary"):
                    usd, eur, fx = _price_both(exit_px, exit_ccy, exit_date)
                    if usd is None:
                        st.error("No pude obtener el EUR/USD de esa fecha; introduce el precio en USD.")
                    else:
                        p = st.session_state.positions[sel_i]
                        ent_eur = p.get("precio_entrada_eur")
                        if ent_eur is None and p.get("precio_entrada"):
                            fx_e = get_fx_rate(p["fecha_entrada"])
                            ent_eur = p["precio_entrada"] / fx_e if fx_e else None
                            p["precio_entrada_eur"] = (round(ent_eur, 4) if ent_eur else None); p["fx_entrada"] = fx_e
                        ratio_eur = (eur / ent_eur) if (eur is not None and ent_eur) else (usd / p["precio_entrada"])
                        p.update({
                            "cerrada": True, "fecha_salida": str(exit_date),
                            "precio_salida": round(usd, 4), "precio_salida_eur": (round(eur, 4) if eur is not None else None),
                            "fx_salida": fx, "moneda_salida": exit_ccy, "motivo_salida": exit_note,
                            "pnl_pct": round((ratio_eur - 1) * 100, 2),
                            "pnl_eur": round(p["capital"] * (ratio_eur - 1), 2),
                        })
                        save_positions(st.session_state.positions)
                        st.success(f"Salida {p['ticker']} guardada: {usd:.2f} $" + (f" = {eur:.2f} €" if eur is not None else "") + f"  → P&L {p['pnl_eur']:+.2f} € ({p['pnl_pct']:+.1f}%)")
                        st.rerun()

    # ── Tabla de posiciones abiertas ─────────────────────────────────────────
    if not open_positions:
        st.info("No tienes posiciones abiertas. Usa el formulario de arriba para agregar una.")
    else:
        pos_rows = []
        for i, pos in open_positions:
            t     = pos["ticker"]
            epx   = pos["precio_entrada"]
            cap   = pos["capital"]
            edate = pd.Timestamp(pos["fecha_entrada"]).date()
            days  = (TODAY - edate).days

            cpx   = cur_prices.get(t, {}).get("px")
            yht   = get_yhat_at(sdata[t]["fc_dict"], TODAY_TS) if t in sdata else None

            pnl_pct    = (cpx / epx - 1) * 100 if cpx else None
            pnl_eur    = cap * (cpx / epx - 1) if cpx else None
            sl_px      = epx * (1 - sl_pct)
            # salida anclada a yhat (como la pestaña de señales y el backtest 15), no al precio de entrada
            tp_px      = yht[0] * (1 + exit_dev) if yht else None
            dist_tp    = (tp_px / cpx - 1) * 100 if (tp_px and cpx) else None
            dist_sl    = (sl_px / cpx - 1) * 100 if cpx else None
            dev_yhat   = (cpx / yht[0] - 1) * 100 if (cpx and yht) else None

            if days >= max_hold:
                accion = "⏰ VENCER PRONTO"
            elif cpx and cpx <= sl_px:
                accion = "🛑 SL ACTIVADO"
            elif tp_px and cpx and cpx >= tp_px:
                accion = "✅ TP ALCANZADO"
            else:
                accion = "⏳ Activa"

            pos_rows.append({
                "#":           i,
                "Ticker":      t,
                "Entrada":     edate,
                "Precio Ent.": epx,
                "Prec. Ent. €": pos.get("precio_entrada_eur"),
                "Precio Hoy":  round(cpx, 2) if cpx else None,
                "P&L%":        round(pnl_pct, 1) if pnl_pct is not None else None,
                "P&L EUR":     round(pnl_eur, 2) if pnl_eur is not None else None,
                "Dias":        days,
                "Dev yhat%":   round(dev_yhat, 1) if dev_yhat is not None else None,
                "SL Ref $":    round(sl_px, 2),
                "TP Ref $":    round(tp_px, 2) if tp_px else None,
                "Dist TP%":    round(dist_tp, 1) if dist_tp is not None else None,
                "Dist SL%":    round(dist_sl, 1) if dist_sl is not None else None,
                "Estado":      accion,
                "Nota":        pos.get("nota", ""),
            })

        df_pos = pd.DataFrame(pos_rows)

        # P&L agregado (solo abiertas)
        total_invested = sum(p["capital"] for _, p in open_positions)
        total_pnl_eur  = df_pos["P&L EUR"].dropna().sum()
        total_pnl_pct  = total_pnl_eur / total_invested * 100 if total_invested > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capital invertido", f"€{total_invested:,.0f}")
        m2.metric("P&L latente", f"€{total_pnl_eur:+,.2f}", f"{total_pnl_pct:+.1f}%")
        m3.metric("Posiciones abiertas", len(open_positions))
        m4.metric("Con TP alcanzado", len(df_pos[df_pos["Estado"].str.contains("TP")]))

        st.dataframe(
            df_pos.drop(columns=["#"]),
            use_container_width=True,
            height=350,
            hide_index=True,
            column_config={
                "Ticker":      st.column_config.TextColumn("Ticker", width=70),
                "Entrada":     st.column_config.DateColumn("Entrada", width=95),
                "Precio Ent.": st.column_config.NumberColumn("Prec. Ent. $", format="$%.2f"),
                "Prec. Ent. €": st.column_config.NumberColumn("Prec. Ent. €", format="€%.2f"),
                "Precio Hoy":  st.column_config.NumberColumn("Prec. Hoy $", format="$%.2f"),
                "P&L%":        st.column_config.NumberColumn("P&L%", format="%.1f%%"),
                "P&L EUR":     st.column_config.NumberColumn("P&L EUR", format="€%.2f"),
                "Dias":        st.column_config.NumberColumn("Dias", format="%d"),
                "Dev yhat%":   st.column_config.NumberColumn("Dev yhat%", format="%.1f%%"),
                "SL Ref $":    st.column_config.NumberColumn("SL Ref $", format="$%.2f"),
                "TP Ref $":    st.column_config.NumberColumn("TP Ref $", format="$%.2f"),
                "Dist TP%":    st.column_config.NumberColumn("Dist TP%", format="%.1f%%"),
                "Dist SL%":    st.column_config.NumberColumn("Dist SL%", format="%.1f%%"),
                "Estado":      st.column_config.TextColumn("Estado", width=150),
                "Nota":        st.column_config.TextColumn("Nota", width=180),
            },
        )

    # ── Mis salidas (posiciones cerradas) ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Mis salidas")
    if not closed_positions:
        st.caption("Aun no has registrado ninguna venta.")
    else:
        df_cl = pd.DataFrame([{
            "Ticker": p["ticker"], "Entrada": pd.Timestamp(p["fecha_entrada"]).date(),
            "Prec. Ent. $": p["precio_entrada"], "Prec. Ent. €": p.get("precio_entrada_eur"),
            "Salida": pd.Timestamp(p["fecha_salida"]).date(),
            "Prec. Sal. $": p["precio_salida"], "Prec. Sal. €": p.get("precio_salida_eur"),
            "Dias": (pd.Timestamp(p["fecha_salida"]) - pd.Timestamp(p["fecha_entrada"])).days,
            "Capital €": p["capital"], "P&L%": p.get("pnl_pct"), "P&L EUR": p.get("pnl_eur"),
            "Motivo": p.get("motivo_salida", ""), "Nota": p.get("nota", ""),
        } for _, p in closed_positions]).sort_values("Salida", ascending=False)
        r1, r2, r3 = st.columns(3)
        r1.metric("Ventas registradas", len(df_cl))
        r2.metric("P&L realizado", f"€{df_cl['P&L EUR'].fillna(0).sum():+,.2f}")
        r3.metric("Acierto", f"{(df_cl['P&L EUR'].fillna(0) > 0).mean()*100:.0f}%")
        st.dataframe(df_cl, use_container_width=True, hide_index=True, column_config={
            "Prec. Ent. $": st.column_config.NumberColumn(format="$%.2f"), "Prec. Ent. €": st.column_config.NumberColumn(format="€%.2f"),
            "Prec. Sal. $": st.column_config.NumberColumn(format="$%.2f"), "Prec. Sal. €": st.column_config.NumberColumn(format="€%.2f"),
            "Capital €": st.column_config.NumberColumn(format="€%.0f"), "P&L%": st.column_config.NumberColumn(format="%.1f%%"),
            "P&L EUR": st.column_config.NumberColumn(format="€%.2f"),
        })

    # Eliminar posiciones (abiertas o cerradas)
    if st.session_state.positions:
        st.markdown("---")
        with st.form("del_pos"):
            del_idx = st.multiselect(
                "Eliminar posiciones",
                options=list(range(len(st.session_state.positions))),
                format_func=lambda i: (f"{st.session_state.positions[i]['ticker']}  "
                                       f"(entrada {st.session_state.positions[i]['fecha_entrada']}"
                                       + (", cerrada" if st.session_state.positions[i].get("cerrada") else "") + ")"),
            )
            if st.form_submit_button("Eliminar seleccionadas"):
                st.session_state.positions = [
                    p for j, p in enumerate(st.session_state.positions) if j not in del_idx
                ]
                save_positions(st.session_state.positions)
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS 6-7 — CARTERA DE PAPEL DE CLAUDE (dos estrategias, un dia cada vez, sin mirar el futuro)
# ═══════════════════════════════════════════════════════════════════════════════
PAPER_FILES = {"bt15": "paper_bt15.json", "combo": "paper_combo.json"}
if "paper" not in st.session_state:
    st.session_state.paper = {k: (load_json(f, None) or paper.new_state(k)) for k, f in PAPER_FILES.items()}

# Avance diario: solo con la ultima sesion COMPLETA (fecha < hoy, para no usar una vela intradia),
# y solo si esa sesion aun no se habia procesado. yhat sale del ajuste de esta sesion, entrenado
# con cierres hasta esa misma fecha. Las ordenes decididas hoy se ejecutan a la apertura siguiente.
_last_bar = max((v["date"] for v in cur_prices.values()), default=None)
_held = set()
for _st in st.session_state.paper.values():
    for _s in _st["slots"]:
        if _s["position"]:
            _held.add(_s["position"]["ticker"])
        if _s["pending"]:
            _held.add(_s["pending"]["ticker"])
_extra = tuple(sorted(_held - set(sdata.keys())))
_all_px = {**cur_prices, **(get_prices(_extra, _buster=px_buster) if _extra else {})}
_paper_msgs = []
if _last_bar and _last_bar < TODAY:
    _day = {}
    for _t, _info in _all_px.items():
        if _info["date"] != _last_bar:
            continue
        _yh = get_yhat_at(sdata[_t]["fc_dict"], pd.Timestamp(_last_bar)) if _t in sdata else None
        _day[_t] = dict(open=_info.get("open"), close=_info["px"], prev_close=_info.get("prev_px"),
                        yhat=(_yh[0] if _yh else None), screen=(_t in sdata))
    for _k, _state in st.session_state.paper.items():
        if _state["last_date"] is None or date.fromisoformat(_state["last_date"]) < _last_bar:
            paper.step(_state, _last_bar, _day)
            save_json(PAPER_FILES[_k], _state, f"Cartera de papel {_k}: sesion {_last_bar}")
            _paper_msgs.append(f"{_k}: procesada la sesion del {_last_bar}")


def render_paper(state, key):
    cfg = paper.STRATEGIES[key]
    s = paper.summary(state)
    st.subheader(f"Cartera de papel · {cfg['name']}")
    st.caption(
        f"Entrada: precio entre {cfg['entry_dev']:.0f}% y {cfg['max_dev']:.0f}% por debajo de yhat"
        + (f" **y** caida del dia >= {cfg['drop']:.0f}%" if cfg["drop"] else "")
        + f", con screening. Salida (en este orden): {cfg['hold']} dias → stop loss {cfg['sl']:.0f}% → precio >= yhat +{cfg['exit_dev']:.0f}%. "
        f"Capital {paper.CAPITAL:.0f} $ en {paper.SLOTS} slots, {paper.FEE:.0f} $ por operacion. "
        "Cada decision usa solo los datos de ese dia y se ejecuta a la apertura del dia siguiente; "
        "la app avanza un dia cada vez que se abre con una sesion nueva completa."
    )
    if _paper_msgs:
        st.info(" · ".join(_paper_msgs))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Valor cartera", f"${s['equity']:,.2f}", f"{s['ret']:+.1%}")
    c2.metric("Ultima sesion", state["last_date"] or "—")
    c3.metric("Operaciones cerradas", s["n_closed"])
    c4.metric("Acierto", f"{s['win_rate']*100:.0f}%" if s["win_rate"] is not None else "—")
    c5.metric("P&L cerrado", f"${s['pnl_closed']:+,.2f}")

    # posiciones abiertas y ordenes pendientes
    rows = []
    for i, slot in enumerate(state["slots"]):
        p, o = slot["position"], slot["pending"]
        if p:
            cpx = _all_px.get(p["ticker"], {}).get("px")
            yh = get_yhat_at(sdata[p["ticker"]]["fc_dict"], TODAY_TS) if p["ticker"] in sdata else None
            rows.append({"Slot": i + 1, "Estado": "abierta" + (f" · orden de venta ({o['reason']})" if o and o["side"] == "sell" else ""),
                         "Ticker": p["ticker"], "Señal": p["signal_date"], "Dev señal%": p["signal_dev"],
                         "Caida señal%": p.get("signal_drop"), "Compra": p["fill_date"], "Precio compra": p["fill_px"],
                         "Precio hoy": cpx, "P&L%": ((cpx / p["fill_px"] - 1) * 100 if cpx else None),
                         "Dias": (TODAY - date.fromisoformat(p["fill_date"])).days,
                         "Vender si >=": (yh[0] * (1 + cfg["exit_dev"] / 100) if yh else None),
                         "Stop loss": p["fill_px"] * (1 - cfg["sl"] / 100)})
        elif o:
            rows.append({"Slot": i + 1, "Estado": "orden de compra pendiente", "Ticker": o["ticker"], "Señal": o["signal_date"],
                         "Dev señal%": o["dev"], "Caida señal%": o.get("drop"), "Compra": None, "Precio compra": None,
                         "Precio hoy": _all_px.get(o["ticker"], {}).get("px"), "P&L%": None, "Dias": None, "Vender si >=": None, "Stop loss": None})
        else:
            rows.append({"Slot": i + 1, "Estado": f"libre ({slot['cash']:.2f} $)", "Ticker": None, "Señal": None, "Dev señal%": None,
                         "Caida señal%": None, "Compra": None, "Precio compra": None, "Precio hoy": None, "P&L%": None, "Dias": None, "Vender si >=": None, "Stop loss": None})
    st.markdown("#### Posiciones abiertas y ordenes")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config={
        "Dev señal%": st.column_config.NumberColumn(format="%.1f%%"), "Caida señal%": st.column_config.NumberColumn(format="%.1f%%"),
        "Precio compra": st.column_config.NumberColumn(format="$%.2f"), "Precio hoy": st.column_config.NumberColumn(format="$%.2f"),
        "P&L%": st.column_config.NumberColumn(format="%.1f%%"), "Vender si >=": st.column_config.NumberColumn(format="$%.2f"),
        "Stop loss": st.column_config.NumberColumn(format="$%.2f")})

    st.markdown("#### Operaciones cerradas")
    if state["closed"]:
        df_c = pd.DataFrame([{"Ticker": c["ticker"], "Señal": c["signal_date"], "Dev señal%": c["signal_dev"], "Caida señal%": c.get("signal_drop"),
                              "Compra": c["fill_date"], "Precio compra": c["fill_px"], "Venta": c["exit_date"], "Precio venta": c["exit_px"],
                              "Motivo": c["exit_reason"], "Dias": c["days"], "P&L%": c["pnl_pct"] * 100, "P&L $": c["pnl_cash"]}
                             for c in state["closed"]]).sort_values("Venta", ascending=False)
        st.dataframe(df_c, use_container_width=True, hide_index=True, column_config={
            "Dev señal%": st.column_config.NumberColumn(format="%.1f%%"), "Caida señal%": st.column_config.NumberColumn(format="%.1f%%"),
            "Precio compra": st.column_config.NumberColumn(format="$%.2f"), "Precio venta": st.column_config.NumberColumn(format="$%.2f"),
            "P&L%": st.column_config.NumberColumn(format="%.1f%%"), "P&L $": st.column_config.NumberColumn(format="$%.2f")})
    else:
        st.caption("Todavia no hay operaciones cerradas.")

    if state["equity"]:
        eq = pd.DataFrame(state["equity"])
        fig = go.Figure(go.Scatter(x=eq["date"], y=eq["equity"], mode="lines", line=dict(color="#4facde", width=2), name="Cartera"))
        fig.add_hline(y=paper.CAPITAL, line_dash="dot", line_color="rgba(200,200,200,0.4)")
        fig.update_layout(height=260, margin=dict(l=40, r=20, t=20, b=30), paper_bgcolor="#0c1118", plot_bgcolor="#131c27",
                          font_color="#c8d6e5", yaxis=dict(tickprefix="$", gridcolor="#1e2d40"), xaxis=dict(gridcolor="#1e2d40"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Diario de decisiones (dia a dia)"):
        for e in reversed(state["log"]):
            st.markdown(f"**{e['date']}**  \n" + "  \n".join(f"· {x}" for x in e["events"]))


with tab_paper1:
    render_paper(st.session_state.paper["bt15"], "bt15")
with tab_paper2:
    render_paper(st.session_state.paper["combo"], "combo")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — COMO FUNCIONA PROPHET
# ═══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.subheader("La estadistica detras de Prophet")
    st.caption(
        "Basado en la documentacion oficial de Meta/Facebook Prophet "
        "(facebook.github.io/prophet) y en el paper original "
        "*Forecasting at Scale* (Taylor & Letham, 2018)."
    )

    st.markdown("### 1. El modelo generativo")
    st.markdown(
        "Prophet es un **modelo de regresion aditivo (o multiplicativo)**, no una red "
        "neuronal ni un ARIMA clasico. Descompone la serie de precios en tres piezas "
        "interpretables por separado, mas un termino de error:"
    )
    st.latex(r"y(t) = g(t) + s(t) + h(t) + \varepsilon_t")
    st.markdown(
        "- **g(t) — Tendencia**: hacia donde va el precio en el largo plazo (curva base).\n"
        "- **s(t) — Estacionalidad**: patrones periodicos (anual, semanal) que se repiten.\n"
        "- **h(t) — Holidays/eventos**: efectos puntuales de fechas especiales (esta app no "
        "los usa, se deja en cero).\n"
        "- **εₜ — Error**: todo lo que el modelo no explica; se asume ruido normal.\n\n"
        "El **yhat** que ves en los charts es la suma `g(t) + s(t) + h(t)` — la 'linea justa' "
        "que Prophet cree que deberia tener el precio ese dia, sin el ruido del dia a dia."
    )

    st.divider()
    st.markdown("### 2. La tendencia g(t): piecewise linear + changepoints")
    st.markdown(
        "Prophet no ajusta una sola linea recta a todo el historico. Coloca automaticamente "
        "**~25 puntos de quiebre potenciales (changepoints)**, distribuidos en el primer 80% "
        "de los datos, donde la pendiente de la tendencia puede cambiar abruptamente (por "
        "ejemplo, cuando una accion pasa de crecer 10%/año a crecer 40%/año tras un buen "
        "resultado trimestral).\n\n"
        "Cuanto puede 'doblarse' la tendencia en cada changepoint se controla con un "
        "**prior tipo Laplace (equivalente a regularizacion L1)** sobre la magnitud de esos "
        "cambios — la mayoria de los changepoints terminan con un cambio de pendiente "
        "practicamente nulo, y solo unos pocos absorben cambios reales de comportamiento. "
        "Ese es exactamente el parametro **'Flexibilidad de tendencia' (changepoint_prior_scale)** "
        "del panel de la izquierda:\n"
        "- **Valor alto** → mas changepoints activos → yhat persigue mas de cerca al precio → "
        "riesgo de sobreajuste (confundir ruido con tendencia real).\n"
        "- **Valor bajo** → tendencia mas rigida/suave → riesgo de ignorar un cambio real de "
        "comportamiento del precio."
    )

    st.divider()
    st.markdown("### 3. La estacionalidad s(t): series de Fourier")
    st.markdown(
        "La estacionalidad se modela como una **suma parcial de series de Fourier** "
        "(senos y cosenos de distintas frecuencias). El *orden* de esa suma (10 para "
        "estacionalidad anual, 3 para semanal, por defecto) define cuantos armonicos se "
        "usan — mas orden = puede capturar patrones estacionales mas complejos, pero con "
        "mas riesgo de sobreajuste, porque cada armonico agrega 2 parametros al modelo.\n\n"
        "El selector **'Seasonality mode'** define como se combina s(t) con la tendencia:\n"
        "- **multiplicative** (recomendado para acciones): el efecto estacional es un "
        "*porcentaje* de la tendencia — si la tendencia crece, la amplitud estacional crece "
        "con ella. Coherente con precios que se mueven en % (crecimiento exponencial).\n"
        "- **additive**: el efecto estacional es un monto fijo en dolares, sin importar el "
        "nivel de precio — tiene mas sentido en series donde la amplitud estacional no "
        "escala con el nivel (ej. temperaturas)."
    )

    st.divider()
    st.markdown("### 4. Intervalos de incertidumbre (la banda sombreada de los charts)")
    st.markdown(
        "Prophet **no** calcula la banda de incertidumbre con una formula cerrada tipo "
        "intervalo de confianza clasico. En su lugar:\n\n"
        "1. Mide la frecuencia y magnitud de los changepoints **historicos**.\n"
        "2. Asume que el futuro tendra cambios de tendencia con esa misma distribucion "
        "estadistica (frecuencia/magnitud).\n"
        "3. Simula muchos futuros posibles con esos cambios aleatorios y calcula los "
        "percentiles de esas simulaciones para dibujar la banda.\n\n"
        "El slider **'Intervalo de confianza (%)'** (interval_width) define que percentil "
        "se dibuja (80%, 95%, etc.) — por defecto Prophet usa estimacion puntual (MAP) via "
        "optimizacion, no muestreo Bayesiano completo (MCMC), salvo que se active "
        "explicitamente `mcmc_samples`. **Esta banda es solo visual: no participa en el "
        "calculo de las señales de compra/venta de esta app**, que se basan unicamente en "
        "la desviacion del precio respecto a yhat."
    )

    st.divider()
    st.markdown("### 5. Que metricas añade esta app (no son de Prophet)")
    st.markdown(
        "Prophet solo entrega `yhat`, `yhat_lower`, `yhat_upper`. Las metricas de la pestaña "
        "**🔍 Screening** las calcula esta app comparando el yhat ajustado contra el precio "
        "historico real:\n\n"
        "- **R²**: correlacion al cuadrado entre precio real y yhat en el periodo de "
        "entrenamiento. Mide que tan bien Prophet 'explica' el movimiento historico del "
        "precio (1 = ajuste perfecto).\n"
        "- **Sigma (σ)**: desviacion estandar de los residuos `(precio − yhat) / yhat` — el "
        "'ruido' que el modelo no logra explicar. Sigma alto implica que la desviacion de "
        "hoy respecto a yhat es menos informativa (puede ser solo ruido, no una señal real).\n"
        "- **Growth anualizado**: tasa de crecimiento compuesta implicita en el yhat entre el "
        "primer y ultimo dia proyectado — sirve para descartar acciones estancadas o con "
        "crecimiento no sostenible.\n"
        "- **ADR (Average Daily Range)**: `(maximo − minimo) / cierre`, promedio de los "
        "ultimos 90 dias — mide volatilidad intradiaria, independiente de Prophet.\n\n"
        "La estrategia de esta app es de **reversion a la media**: usa yhat como 'valor "
        "justo' de referencia, y genera señales cuando el precio se desvia demasiado de "
        "esa referencia (🟢 por debajo = posible compra, 🔴 por arriba = posible venta), "
        "filtrando de antemano las acciones donde Prophet ajusta mal (R² bajo, sigma alto) "
        "para que esa referencia sea confiable."
    )

    st.info(
        "📖 Documentacion completa: "
        "[facebook.github.io/prophet/docs](https://facebook.github.io/prophet/docs/quick_start.html)"
    )
