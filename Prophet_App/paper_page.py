"""Genera paper_page.html: replica publica de la app (artifact) con todas las pestañas.

Los datos (ajustes Prophet, bandas, proyecciones, metricas de screening, posiciones, carteras de
papel) se precomputan en daily_update.py y se embeben en la pagina; los sliders de estrategia y
screening recalculan en el navegador sobre esos datos. Plotly (cdnjs) dibuja las graficas.
"""
import html, json, re


def _versions_html(md):
    if not md:
        return "<p class='muted'>Sin VERSIONES.md</p>"
    rows = [l for l in md.splitlines() if l.startswith("|") and not l.startswith("|---")]
    if not rows:
        return "<pre>" + html.escape(md) + "</pre>"
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    body = []
    for r in rows[1:]:
        cells = [c.strip() for c in r.strip("|").split("|")]
        cells = [re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(c)).replace("`", "") for c in cells]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return ("<div class='tbl'><table><thead><tr>" + "".join(f"<th title='clic para ordenar'>{html.escape(h)}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


DOCS_HTML = """
<h3>1. El modelo</h3><p>Prophet es un modelo de regresion aditivo (o multiplicativo): descompone el precio en tendencia g(t) + estacionalidad s(t) + error. El <b>yhat</b> que ves es g(t)+s(t): la "linea justa" que el modelo cree que deberia tener el precio ese dia, sin el ruido diario.</p>
<h3>2. Tendencia: lineal a trozos con changepoints</h3><p>Prophet coloca ~25 puntos de quiebre potenciales en el primer 80 % de los datos donde la pendiente puede cambiar. Cuanto puede doblarse se controla con <code>changepoint_prior_scale</code> (0,03 aqui): alto = tendencia pegada al precio (sobreajuste), bajo = tendencia rigida.</p>
<h3>3. Estacionalidad: series de Fourier</h3><p>La estacionalidad anual se modela como suma de senos y cosenos. En modo <b>multiplicative</b> el efecto estacional es un porcentaje de la tendencia, coherente con precios que se mueven en %.</p>
<h3>4. La banda sombreada</h3><p>Prophet simula futuros con cambios de tendencia parecidos a los del pasado y dibuja los percentiles (IC 95 %). Es solo visual: las señales dependen solo de la desviacion del precio respecto a yhat.</p>
<h3>5. Metricas de esta app (no son de Prophet)</h3><p><b>R²</b>: correlacion al cuadrado entre precio y yhat en el entrenamiento. <b>Sigma</b>: desviacion estandar de (precio − yhat)/yhat. <b>Crecimiento</b>: tasa anual implicita en yhat. <b>ADR</b>: (maximo − minimo)/cierre medio de 90 dias. La estrategia es de reversion a la media: compra cuando el precio cae bajo yhat y vende cuando vuelve.</p>
<h3>6. Lo que dice el backtest 15 (2021-2026)</h3><p>Ni la config antigua (3 / TP 20 / SL 15 / 15 d) ni la actual (4 / yhat +7 / SL 7 / 60 d) baten a QQQ de forma clara en cartera con 1.000 $: la ventaja por operacion es de decimas y las comisiones la consumen. Las carteras de papel de las pestañas 🧪 sirven para comprobarlo en tiempo real, sin trampas.</p>
"""


def build(data):
    """data: dict con D, version, tickers{...}, screening, strategy, capital, positions, paper{bt15,combo}, cfgs, versions_md."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    D = data["D"]
    versions_html = _versions_html(data.get("versions_md", ""))
    return f"""<title>Prophet Trader</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.0/plotly.min.js"></script>
<style>
:root {{ --bg:#f3f4f1; --panel:#fff; --fg:#1d2229; --muted:#5d6673; --line:#d9dcd6; --accent:#b86a00; --accent-soft:#fff1dc; --ok:#2c7a4b; --ok-soft:#e2f1e7; --bad:#b23a3a; --bad-soft:#f7e2e2; --blue:#2f7fb8;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#15181c; --panel:#1d2127; --fg:#e6e8e3; --muted:#9aa3ad; --line:#2c3139; --accent:#f0a23a; --accent-soft:#3a2a12; --ok:#6fcf93; --ok-soft:#173422; --bad:#ef8a8a; --bad-soft:#3d1c1c; --blue:#4facde; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#15181c; --panel:#1d2127; --fg:#e6e8e3; --muted:#9aa3ad; --line:#2c3139; --accent:#f0a23a; --accent-soft:#3a2a12; --ok:#6fcf93; --ok-soft:#173422; --bad:#ef8a8a; --bad-soft:#3d1c1c; --blue:#4facde; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:14px; line-height:1.45; padding-block:16px 48px;
  padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); }}
.wrap {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-family:var(--mono); font-size:20px; font-weight:600; margin:0; }} .sub {{ color:var(--muted); font-size:13px; margin:2px 0 12px; }}
h2 {{ font-size:17px; margin:18px 0 6px; }} h3 {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:16px 0 6px; }}
.muted, .n {{ color:var(--muted); }} .note {{ font-size:13px; color:var(--muted); max-width:75ch; margin:8px 0; }}
.tabs {{ display:flex; flex-wrap:wrap; gap:4px; border-bottom:1px solid var(--line); position:sticky; top:env(safe-area-inset-top, 0px); background:var(--bg); z-index:5; padding:6px 0; }}
.tab {{ font:600 13px var(--sans); color:var(--muted); background:transparent; border:0; border-bottom:3px solid transparent; padding:6px 10px; cursor:pointer; }}
.tab[aria-selected="true"] {{ color:var(--fg); border-bottom-color:var(--blue); }} .tab:focus-visible {{ outline:2px solid var(--blue); outline-offset:2px; }}
.panel {{ padding-top:8px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:8px; margin:8px 0; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); padding:10px 12px; min-width:0; }}
.kpi .l {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }} .kpi .v {{ font-family:var(--mono); font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; }} .kpi .n {{ font-size:12px; }}
.kpi.spark {{ grid-column:span 2; }}
.controls {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:8px 16px; background:var(--panel); border:1px solid var(--line); padding:10px 12px; margin:8px 0; }}
.ctl label {{ display:flex; justify-content:space-between; font-size:12px; color:var(--muted); }} .ctl label b {{ color:var(--fg); font-family:var(--mono); }}
.ctl input[type=range] {{ width:100%; accent-color:var(--blue); }} select, input[type=date] {{ font:13px var(--sans); color:var(--fg); background:var(--panel); border:1px solid var(--line); padding:5px 8px; }}
.tbl {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line); max-height:70vh; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }} th, td {{ padding:6px 9px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); position:sticky; top:0; background:var(--panel); cursor:pointer; }}
td.num, th.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }} tr:last-child td {{ border-bottom:0; }}
.good {{ color:var(--ok); font-weight:500; }} .poor {{ color:var(--bad); font-weight:500; }} .warn {{ color:var(--accent); font-weight:500; }}
.pill {{ display:inline-block; font:600 11px var(--mono); padding:1px 7px; border-radius:999px; }} .pill.buy {{ background:var(--ok-soft); color:var(--ok); }} .pill.sell {{ background:var(--bad-soft); color:var(--bad); }} .pill.low {{ background:var(--accent-soft); color:var(--accent); }} .pill.neu {{ border:1px solid var(--line); color:var(--muted); }} .pill.knife {{ background:var(--bad); color:#fff; }}
.banner {{ background:var(--ok-soft); color:var(--ok); border:1px solid var(--ok); padding:8px 12px; margin:8px 0; font-size:13px; }}
a.tk {{ color:var(--fg); text-decoration:none; border-bottom:1px dotted var(--blue); }} a.tk:hover {{ color:var(--blue); }}
.btn {{ font:600 13px var(--sans); color:var(--fg); background:var(--panel); border:1px solid var(--line); padding:7px 12px; cursor:pointer; white-space:nowrap; }}
.btn:hover {{ border-color:var(--blue); color:var(--blue); }} .btn:focus-visible {{ outline:2px solid var(--blue); outline-offset:2px; }}
.chart {{ background:var(--panel); border:1px solid var(--line); min-height:420px; }}
details {{ margin-top:10px; }} summary {{ cursor:pointer; color:var(--muted); font-size:13px; }} .day {{ font-size:13px; margin:8px 0; }} .day ul {{ margin:2px 0 0 18px; padding:0; }}
.docs p {{ max-width:75ch; }} code {{ font-family:var(--mono); font-size:12px; }}
.foot {{ margin-top:24px; font-family:var(--mono); font-size:12px; color:var(--muted); }}
</style>
<div class="wrap">
  <h1>📈 Prophet Trader</h1>
  <p class="sub">Replica publica de la app · ultima sesion <b>{html.escape(D)}</b> · v{html.escape(str(data.get('version', '?')))} · se actualiza cada dia laborable tras el cierre de EE.UU. Ventana de entrenamiento 1 año, changepoint 0,03, multiplicative (fijos; el resto de controles recalcula aqui mismo).</p>
  <div class="tabs" role="tablist" id="tabs"></div>
  <div id="panels"></div>
  <div class="foot">Fuente: Yahoo Finance (cierres ajustados). Repositorio: JuanMEcheverri/Prophet_App. Simulacion sin dinero real; no es una recomendacion de inversion.</div>
</div>
<script id="data" type="application/json">{payload.replace("</", "<\\/")}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const VERSIONS_HTML = {json.dumps(versions_html)};
const DOCS_HTML = {json.dumps(DOCS_HTML)};
const T = DATA.tickers, D = DATA.D, NM = DATA.names || {{}};
const lbl = tk => (tk && NM[tk]) ? `${{tk}} · ${{NM[tk]}}` : (tk || '—');
const lblHtml = tk => tk ? `<a class="tk" href="#chart" data-tk="${{tk}}" title="Ver la grafica de ${{tk}}"><b>${{tk}}</b>${{NM[tk] ? ` <span class="muted">${{esc(NM[tk])}}</span>` : ''}}</a>` : '—';
document.addEventListener('click', e => {{
  const a = e.target.closest('a.tk'); if (!a) return;
  e.preventDefault();
  const tk = a.getAttribute('data-tk'), sel = document.getElementById('chart_t');
  if (sel && Array.from(sel.options).some(o => o.value === tk)) sel.value = tk;
  const q = document.getElementById('chart_q'); if (q) q.value = '';
  show('chart');
  window.scrollTo({{top: 0, behavior: 'smooth'}});
}});
const S = {{...DATA.strategy}};      // entry, exit, sl, hold, max_dev (en %)
const SC = {{...DATA.screening}};    // min_r2, max_sigma, min_growth, max_growth, max_adr
const fmt = (x, d=2) => (x==null||isNaN(x)) ? '—' : x.toLocaleString('es-ES', {{minimumFractionDigits:d, maximumFractionDigits:d}});
const pct = (x, d=1) => (x==null||isNaN(x)) ? '—' : (x>0?'+':'') + fmt(x, d) + ' %';
const usd = x => (x==null||isNaN(x)) ? '—' : fmt(x, 2) + ' $';
const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let PRESET = 'manual', DROP_REQ = null;   // preset activo en Señales y exigencia de caida del dia (combinado)
function setSliderUI(id, val, suffix) {{
  const el = document.getElementById(id); if (!el) return;
  el.value = val; const v = document.getElementById(id + '_v'); if (v) v.textContent = val + (suffix || '');
}}
function applyPreset(key) {{
  const cfg = DATA.cfgs[key], profs = (DATA.paper[key] && DATA.paper[key].screening) || [];
  const p = profs.length ? profs[profs.length - 1] : null;      // perfil vigente para la proxima decision
  S.entry = cfg.entry_dev; S.exit = cfg.exit_dev; S.sl = cfg.sl; S.hold = cfg.hold; S.max_dev = cfg.max_dev;
  setSliderUI('s_entry', S.entry, ' %'); setSliderUI('s_exit', S.exit, ' %'); setSliderUI('s_sl', S.sl, ' %'); setSliderUI('s_hold', S.hold, ' d'); setSliderUI('s_maxdev', S.max_dev, ' %');
  if (p) {{
    SC.min_r2 = p.min_r2; SC.max_sigma = Math.round(p.max_sigma * 100); SC.min_growth = Math.round(p.min_growth * 100); SC.max_growth = Math.round(p.max_growth * 100); SC.max_adr = Math.round(p.max_adr * 100);
    setSliderUI('c_r2', SC.min_r2, ''); setSliderUI('c_sig', SC.max_sigma, ' %'); setSliderUI('c_gmin', SC.min_growth, ' %'); setSliderUI('c_gmax', SC.max_growth, ' %'); setSliderUI('c_adr', SC.max_adr, ' %');
  }}
  DROP_REQ = cfg.drop || null; PRESET = key;
  renderScreening(); renderSignals(); renderProj(); renderPositions(); if (!document.getElementById('panel_chart').hidden) renderChart();
}}
const SUFFIXES = ['', '_s'];   // la misma fila de presets aparece en Señales ('') y en Screening ('_s')
function presetRow(sfx) {{
  return `<div class="controls" style="grid-template-columns:repeat(auto-fit, minmax(150px, max-content)) 1fr;align-items:center"><button type="button" class="btn" id="pre_bt15${{sfx}}">Cuadrar con Backtest 15</button><button type="button" class="btn" id="pre_combo${{sfx}}">Cuadrar con Combinado</button><button type="button" class="btn" id="pre_u1${{sfx}}">Mi preset 1</button><button type="button" class="btn" id="pre_u2${{sfx}}">Mi preset 2</button><button type="button" class="btn" id="pre_u3${{sfx}}">Mi preset 3</button><span id="preset_badge${{sfx}}" class="muted" style="font-size:13px"></span></div><p class="note" style="margin:0 0 8px">Los presets fijan a la vez los sliders de estrategia (Señales) <b>y</b> de screening. "Mi preset": <b>clic</b> = cargar (o guardar si esta vacio) · <b>mantener pulsado ~1 s</b> = sobrescribir con los valores actuales. <span id="preset_store${{sfx}}"></span></p>`;
}}
function presetBadge() {{
  const name = PRESET === 'manual' ? null : (DATA.cfgs[PRESET] ? DATA.cfgs[PRESET].name : PRESET);
  const txt = PRESET === 'manual' ? 'Ajuste manual (sliders)' : `Cuadrado con <b>${{esc(name)}}</b>` + (DROP_REQ ? ` · compra solo si la caida del dia es ≤ −${{DROP_REQ}} %` : '');
  SUFFIXES.forEach(sfx => {{ const b = document.getElementById('preset_badge' + sfx); if (b) b.innerHTML = txt; }});
}}
// ---- Presets del usuario (3 huecos): clic = cargar; mantener pulsado ~0,8 s = guardar/sobrescribir.
// Se guardan en la base de datos del artifact (coleccion 'presets', solo el dueño escribe), compartida entre
// dispositivos y legible por Claude; si la base de datos no esta disponible (sin sesion, sin permiso), en este navegador.
const USER_KEY = 'prophet_user_presets_v1';
let DBP = null, DB_PRESETS = {{}}, DB_STATE = 'local';
function localPresets() {{ try {{ return JSON.parse(localStorage.getItem(USER_KEY) || '{{}}'); }} catch (e) {{ return {{}}; }} }}
// Vista combinada: lo guardado en la base de datos manda; si un hueco esta vacio alli pero existe en este
// navegador (presets guardados antes de activar la base de datos), se muestra el local y se migra a la base de datos.
function userPresets() {{
  const local = localPresets();
  if (!DBP) return local;
  const merged = {{...local, ...DB_PRESETS}};
  return merged;
}}
let MIGRATED = false;
function migrateLocalToDb() {{
  if (!DBP || MIGRATED) return;
  MIGRATED = true;
  const local = localPresets();
  Object.keys(local).forEach(n => {{
    if (!DB_PRESETS[n]) DBP.doc('presets/' + n).set(local[n]).then(() => {{ DB_PRESETS[n] = local[n]; refreshUserButtons(); }}).catch(() => {{}});
  }});
}}
function localSave(n, body) {{ const all = localPresets(); all[n] = body; try {{ localStorage.setItem(USER_KEY, JSON.stringify(all)); }} catch (e) {{}} }}
function saveUserPreset(n) {{
  const body = {{S: {{...S}}, SC: {{...SC}}, drop: DROP_REQ, saved: D}};
  PRESET = 'Mi preset ' + n; presetBadge();
  localSave(n, body);   // copia local siempre (respaldo si la base de datos no responde)
  if (DBP) {{
    DB_PRESETS[n] = body; refreshUserButtons(n, 'guardando…');
    DBP.doc('presets/' + n).set(body).then(() => refreshUserButtons(n, 'guardado ✓')).catch(() => {{ DB_STATE = 'local'; refreshUserButtons(n, 'guardado solo en este navegador'); }});
  }} else {{ refreshUserButtons(n, 'guardado en este navegador'); }}
}}
async function initPresetsDb() {{
  try {{
    if (!window.claude || typeof claude.use !== 'function') return;
    const db = await claude.use('db'); if (!db) return;
    db.collection('presets').onSnapshot(snap => {{
      DBP = db; DB_STATE = 'db'; DB_PRESETS = {{}};
      snap.docs.forEach(d => {{ if (d.exists) DB_PRESETS[d.id] = d.data(); }});
      refreshUserButtons();
      if (!snap.metadata.fromCache) migrateLocalToDb();
    }}, err => {{ DBP = null; DB_STATE = 'local'; refreshUserButtons(); }});
  }} catch (e) {{}}
}}
function loadUserPreset(n) {{
  const p = userPresets()[n]; if (!p) return false;
  Object.assign(S, p.S); Object.assign(SC, p.SC); DROP_REQ = p.drop == null ? null : p.drop;
  setSliderUI('s_entry', S.entry, ' %'); setSliderUI('s_exit', S.exit, ' %'); setSliderUI('s_sl', S.sl, ' %'); setSliderUI('s_hold', S.hold, ' d'); setSliderUI('s_maxdev', S.max_dev, ' %');
  setSliderUI('c_r2', SC.min_r2, ''); setSliderUI('c_sig', SC.max_sigma, ' %'); setSliderUI('c_gmin', SC.min_growth, ' %'); setSliderUI('c_gmax', SC.max_growth, ' %'); setSliderUI('c_adr', SC.max_adr, ' %');
  PRESET = 'Mi preset ' + n;
  renderScreening(); renderSignals(); renderProj(); renderPositions(); if (!document.getElementById('panel_chart').hidden) renderChart();
  return true;
}}
function refreshUserButtons(flashN, flashText) {{
  const all = userPresets();
  [1, 2, 3].forEach(n => SUFFIXES.forEach(sfx => {{
    const b = document.getElementById('pre_u' + n + sfx); if (!b) return;
    const p = all[n];
    b.textContent = (flashN === n && flashText) ? `Mi preset ${{n}}: ${{flashText}}` : (p ? `Mi preset ${{n}} ✓` : `Mi preset ${{n}} (vacio)`);
    const st = document.getElementById('preset_store' + sfx); if (st) st.textContent = DB_STATE === 'db' ? 'Presets guardados en la base de datos del artifact (compartidos entre tus dispositivos).' : 'Presets guardados solo en este navegador (base de datos no disponible: entra con tu cuenta para compartirlos entre dispositivos).';
    b.title = p ? `Guardado el ${{p.saved}}: entrada ${{p.S.entry}} % · salida +${{p.S.exit}} % · SL ${{p.S.sl}} % · ${{p.S.hold}} d · cap ${{p.S.max_dev}} % · screening R² ${{p.SC.min_r2}} / sigma ${{p.SC.max_sigma}} % / crec. ${{p.SC.min_growth}}..${{p.SC.max_growth}} % / ADR ${{p.SC.max_adr}} %${{p.drop ? ' · caida ≥ ' + p.drop + ' %' : ''}}. Clic = cargar · mantener pulsado = sobrescribir` : 'Vacio. Clic o mantener pulsado = guardar los sliders actuales (estrategia + screening)';
  }}));
  if (flashN) setTimeout(() => refreshUserButtons(), 1500);
}}
function bindUserButtons() {{
  [1, 2, 3].forEach(n => SUFFIXES.forEach(sfx => {{
    const b = document.getElementById('pre_u' + n + sfx); if (!b) return;
    let timer = null, held = false;
    const start = e => {{ held = false; timer = setTimeout(() => {{ held = true; saveUserPreset(n); }}, 800); }};
    const end = e => {{ if (timer) clearTimeout(timer); timer = null; if (held) return; if (!loadUserPreset(n)) saveUserPreset(n); }};
    b.addEventListener('pointerdown', start); b.addEventListener('pointerup', end);
    b.addEventListener('pointerleave', () => {{ if (timer) clearTimeout(timer); timer = null; }});
    b.addEventListener('contextmenu', e => e.preventDefault());
  }}));
  SUFFIXES.forEach(sfx => {{
    const b1 = document.getElementById('pre_bt15' + sfx), b2 = document.getElementById('pre_combo' + sfx);
    if (b1) b1.onclick = () => applyPreset('bt15');
    if (b2) b2.onclick = () => applyPreset('combo');
  }});
  refreshUserButtons();
}}

function screenReason(t, sc) {{
  const f = [];
  if (t.r2 == null) return 'sin ajuste';
  if (t.r2 < sc.min_r2) f.push(`R² ${{fmt(t.r2,2)}} < ${{sc.min_r2}}`);
  if (t.sigma*100 > sc.max_sigma) f.push(`sigma ${{fmt(t.sigma*100,1)}} % > ${{sc.max_sigma}} %`);
  if (t.growth*100 < sc.min_growth) f.push(`crecimiento ${{fmt(t.growth*100,0)}} % < ${{sc.min_growth}} %`);
  if (t.growth*100 > sc.max_growth) f.push(`crecimiento ${{fmt(t.growth*100,0)}} % > ${{sc.max_growth}} %`);
  if (t.adr*100 > sc.max_adr) f.push(`ADR ${{fmt(t.adr*100,1)}} % > ${{sc.max_adr}} %`);
  return f.length ? f.join('; ') : 'OK';
}}
const passes = (t, sc) => screenReason(t, sc) === 'OK';
function classify(dev, s) {{
  if (dev <= -s.max_dev) return ['⚠️ MUY BAJO', 'low'];
  if (dev <= -s.entry) return ['🟢 COMPRAR', 'buy'];
  if (dev >= s.exit) return ['🔴 SOBRE COMPRADO', 'sell'];
  return ['⚪ NEUTRAL', 'neu'];
}}
function slider(id, label, min, max, step, val, suffix='') {{
  return `<div class="ctl"><label for="${{id}}">${{label}} <b id="${{id}}_v">${{val}}${{suffix}}</b></label><input type="range" id="${{id}}" min="${{min}}" max="${{max}}" step="${{step}}" value="${{val}}"></div>`;
}}
const COLHELP = {{
  'Ticker': 'Simbolo en bolsa y nombre de la empresa', 'Precio': 'Precio de la operacion o ultimo cierre (USD)', 'Cierre': 'Ultimo cierre (USD)',
  'yhat': 'Precio "justo" que estima Prophet para ese dia (tendencia + estacionalidad)', 'Dev %': 'Precio / yhat − 1. Negativo = por debajo del precio justo',
  'Bajo/sobre yhat': 'Precio / yhat − 1 en el dia de la señal', 'Bajo yhat': 'Cuanto estaba el precio por debajo de yhat el dia de la señal',
  'Caida dia': 'Cierre de hoy frente al cierre anterior', 'Caida del dia': 'Cierre de hoy frente al cierre anterior',
  'Señal': 'COMPRAR si Dev esta entre −cap y −entrada; SOBRE COMPRADO si Dev ≥ salida; MUY BAJO si cae mas que el cap; CUCHILLO CAYENDO si la caida de HOY supera el stop loss (comprar ayer ya habria saltado el SL)',
  'Thr compra': 'Umbral de compra: yhat × (1 − entrada %)', 'Thr venta': 'Umbral de venta: yhat × (1 + salida %)',
  'SL ref': 'Precio al que saltaria el stop loss si comprases hoy', 'Dist TP': 'Distancia del precio actual al umbral de venta', 'Dist SL': 'Distancia del precio actual al stop loss',
  'R²': 'Que tan bien explica Prophet el precio del ultimo año (1 = perfecto)', 'Pasa': 'Si la accion cumple todos los filtros del screening',
  'Sigma': 'Ruido: desviacion tipica de (precio − yhat)/yhat en el entrenamiento', 'Crec./año': 'Crecimiento anual implicito en la tendencia de Prophet',
  'ADR': 'Rango diario medio (max − min)/cierre de los ultimos 90 dias: volatilidad intradia', 'Motivo': 'Por que no pasa el screening / por que se vendio',
  'Fecha': 'Dia futuro al que corresponde el yhat', 'Dias': 'Dias naturales', 'yhat lo': 'Limite inferior de la banda de incertidumbre (IC 95 %)', 'yhat hi': 'Limite superior de la banda (IC 95 %)',
  'Precio hoy': 'Ultimo cierre (USD)', 'Entrada': 'Fecha de compra', 'Prec. ent. $': 'Precio de compra por accion en USD', 'Prec. ent. €': 'Precio de compra por accion en EUR (al cambio de ese dia)',
  'Prec. hoy $': 'Ultimo cierre en USD', 'P&L %': 'Ganancia o perdida en porcentaje', 'P&L €': 'Ganancia o perdida en euros sobre el capital invertido', 'P&L': 'Ganancia o perdida en porcentaje',
  'P&L $': 'Ganancia o perdida en dolares', 'TP ref (yhat)': 'Objetivo de venta: yhat × (1 + salida %)', 'Estado': 'Situacion de la posicion segun las reglas', 'Nota': 'Nota libre',
  'Slot': 'Hueco de la cartera (maximo 3 a la vez)', 'Acciones': 'Numero de acciones enteras', 'Compra': 'Fecha de compra (apertura siguiente a la señal)', 'Precio compra': 'Precio de compra por accion (USD, con 0,05 % de deslizamiento)',
  'Invertido': 'Dinero puesto en la posicion, comision incluida', 'Vender si ≥': 'Precio al que la regla vende: yhat × (1 + salida %)', 'Stop loss': 'Precio al que la regla vende por perdida',
  'Dias en zona de compra': 'Cuantos dias del horizonte el precio de hoy quedaria entre −cap y −entrada frente al yhat de ese dia', 'Primer dia': 'Primer dia del horizonte en que entraria en zona de compra', 'Dev % hoy': 'Precio de hoy / yhat de hoy − 1', 'yhat hoy': 'Precio justo estimado para hoy', 'Señal hoy': 'Señal con el yhat de hoy',
  'Plataforma': 'Broker donde se hizo la operacion (N26, Degiro...)', 'Cantidad': 'Numero de acciones (admite fracciones)',
  'Venta': 'Fecha de venta', 'Salida': 'Fecha de venta', 'Prec. sal. $': 'Precio de venta por accion en USD', 'Capital €': 'Capital invertido en euros', 'Pasa screening': 'Si cumple los filtros del screening', 'Version': 'Version de la app', 'Resumen': 'Que cambio',
}};
function parseCell(td) {{
  const s = (td.textContent || '').trim();
  if (s === '—' || s === '') return {{n: null, s}};
  const num = s.replace(/[$€%+\\s]/g, '').replace(/\\./g, '').replace(',', '.');
  const n = parseFloat(num);
  return {{n: isNaN(n) || !/^-?\\d/.test(num) ? null : n, s}};
}}
document.addEventListener('click', e => {{
  const th = e.target.closest('.tbl th'); if (!th) return;
  const table = th.closest('table'), idx = Array.from(th.parentNode.children).indexOf(th);
  const asc = th.getAttribute('data-sort') !== 'asc';
  Array.from(table.querySelectorAll('thead th')).forEach(h => {{ h.removeAttribute('data-sort'); h.textContent = h.textContent.replace(/ [▲▼]$/, ''); }});
  th.setAttribute('data-sort', asc ? 'asc' : 'desc'); th.textContent += asc ? ' ▲' : ' ▼';
  const tb = table.querySelector('tbody'), rows = Array.from(tb.rows);
  rows.sort((a, b) => {{
    const A = parseCell(a.cells[idx]), B = parseCell(b.cells[idx]);
    if (A.n != null && B.n != null) return asc ? A.n - B.n : B.n - A.n;
    if (A.n == null && B.n != null) return 1; if (B.n == null && A.n != null) return -1;
    return asc ? A.s.localeCompare(B.s) : B.s.localeCompare(A.s);
  }});
  rows.forEach(r => tb.appendChild(r));
}});
function table(id, headers, rows) {{
  const th = headers.map(h => `<th class="${{h[1]?'num':''}}" title="${{esc(COLHELP[h[0]] || h[0])}} · clic para ordenar" data-k="${{h[2]||''}}">${{esc(h[0])}}</th>`).join('');
  const body = rows.map(r => '<tr>' + r.map((c, i) => {{
    let v = c, cls = headers[i][1] ? 'num' : '';
    if (Array.isArray(c)) {{ v = c[0]; cls += ' ' + (c[1]||''); }}
    return `<td class="${{cls}}">${{v}}</td>`;
  }}).join('') + '</tr>').join('');
  return `<div class="tbl" id="${{id}}"><table><thead><tr>${{th}}</tr></thead><tbody>${{body}}</tbody></table></div>`;
}}
function bindSliders(ids, obj, onchange) {{
  ids.forEach(([id, key, suffix]) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.oninput = () => {{ obj[key] = parseFloat(el.value); document.getElementById(id+'_v').textContent = el.value + (suffix||''); onchange(); }};
  }});
}}

// ---------- Señales ----------
function renderSignals() {{
  const rows = [];
  let nBuy = 0, nSell = 0, nAll = 0;
  const buyList = [];
  for (const [tk, t] of Object.entries(T)) {{
    if (!passes(t, SC) || t.yhat == null || t.close == null) continue;
    nAll++;
    const dev = (t.close / t.yhat - 1) * 100;
    const drop = t.prev ? (t.close / t.prev - 1) * 100 : null;
    let [sig, cls] = classify(dev, S);
    if (cls === 'buy' && DROP_REQ != null && !(drop != null && drop <= -DROP_REQ)) {{ sig = '⚪ NEUTRAL (sin caida ≥ ' + DROP_REQ + ' %)'; cls = 'neu'; }}
    // caida de un solo dia mayor que el stop loss: si hubieras comprado ayer ya habrias saltado; no es un dip, es un cuchillo cayendo
    if (drop != null && drop <= -S.sl) {{ sig = `🔪 CUCHILLO CAYENDO (${{pct(drop)}} hoy, mas que el SL ${{S.sl}} %)`; cls = 'knife'; }}
    if (cls === 'buy') {{ nBuy++; buyList.push(`<b>${{tk}}</b> ${{pct(dev)}}`); }}
    if (cls === 'sell') nSell++;
    const thrB = t.yhat * (1 - S.entry/100), thrS = t.yhat * (1 + S.exit/100), slp = t.close * (1 - S.sl/100);
    rows.push({{dev, cells: [lblHtml(tk), usd(t.close), usd(t.yhat), [pct(dev), dev<0?'poor':'good'], [pct(drop), drop!=null&&drop<=-5?'warn':''],
      `<span class="pill ${{cls}}">${{sig}}</span>`, usd(thrB), usd(thrS), usd(slp), pct((thrS/t.close-1)*100), pct((slp/t.close-1)*100), fmt(t.r2,2)]}});
  }}
  rows.sort((a,b) => a.dev - b.dev);
  const el = document.getElementById('sig_out');
  presetBadge();
  el.innerHTML = `<div class="kpis">
    <div class="kpi"><div class="l">Acciones analizadas</div><div class="v">${{nAll}}</div></div>
    <div class="kpi"><div class="l">En zona de compra 🟢</div><div class="v good">${{nBuy}}</div></div>
    <div class="kpi"><div class="l">Sobre compradas 🔴</div><div class="v poor">${{nSell}}</div></div>
    <div class="kpi"><div class="l">Neutras ⚪</div><div class="v">${{nAll-nBuy-nSell}}</div></div></div>` +
    (buyList.length ? `<div class="banner">${{buyList.length}} acciones en zona de compra: ${{buyList.join(' · ')}}</div>` : '') +
    table('sig_tbl', [['Ticker',0],['Precio',1],['yhat',1],['Dev %',1],['Caida dia',1],['Señal',0],['Thr compra',1],['Thr venta',1],['SL ref',1],['Dist TP',1],['Dist SL',1],['R²',1]], rows.map(r=>r.cells)) +
    `<p class="note">Dev % = precio / yhat − 1. Señal de compra si Dev entre −${{S.max_dev}} % y −${{S.entry}} %. Caida del dia = cierre vs cierre anterior; la estrategia combinada exige ademas ≤ −5 %. <b>🔪 Cuchillo cayendo</b>: la caida de hoy supera el stop loss (${{S.sl}} %): quien hubiera comprado ayer ya habria salido con perdida; mejor esperar a que deje de caer.</p>`;
}}
// ---------- Screening ----------
function renderScreening() {{
  const rows = [];
  let nPass = 0;
  for (const [tk, t] of Object.entries(T)) {{
    const r = screenReason(t, SC), ok = r === 'OK';
    if (ok) nPass++;
    rows.push({{ok, r2: t.r2||0, cells: [lblHtml(tk), ok ? '<span class="pill buy">pasa</span>' : '<span class="pill neu">no</span>', fmt(t.r2,3), pct(t.sigma==null?null:t.sigma*100), pct(t.growth==null?null:t.growth*100), pct(t.adr==null?null:t.adr*100), r]}});
  }}
  rows.sort((a,b) => (b.ok - a.ok) || (b.r2 - a.r2));
  document.getElementById('scr_out').innerHTML = `<p class="note">✅ ${{nPass}} pasan · ❌ ${{rows.length-nPass}} no pasan · ${{rows.length}} en total. Las señales, graficas y proyecciones usan solo las que pasan.</p>` +
    table('scr_tbl', [['Ticker',0],['Pasa',0],['R²',1],['Sigma',1],['Crec./año',1],['ADR',1],['Motivo',0]], rows.map(r=>r.cells));
  // refrescar selects dependientes
  fillTickerSelects();
}}
function fillTickerSelects() {{
  const all = Object.keys(T).filter(tk => T[tk].fit_d).sort();
  const ok = all.filter(tk => passes(T[tk], SC));
  // Charts: todas las acciones (las que no pasan el screening, marcadas); Proyecciones: solo las que pasan
  const sel = document.getElementById('chart_t');
  if (sel) {{
    const cur = sel.value;
    // Charts: SIEMPRE todas las acciones, en orden alfabetico; las que no pasan el screening, marcadas
    sel.innerHTML = all.map(o => `<option value="${{o}}">${{lbl(o)}}${{ok.includes(o) ? '' : ' (no pasa screening)'}}</option>`).join('');
    sel.value = all.includes(cur) ? cur : all[0];
  }}
  const dl = document.getElementById('chart_list');
  if (dl && !dl.children.length) dl.innerHTML = all.map(o => `<option value="${{lbl(o)}}"></option>`).join('');
  const sel2 = document.getElementById('proj_t2');
  if (sel2) {{
    const cur = sel2.value, opts = ['(todas)'].concat(ok);
    sel2.innerHTML = opts.map(o => `<option value="${{o}}">${{o}}</option>`).join('');
    sel2.value = opts.includes(cur) ? cur : opts[0];
  }}
}}
// ---------- Charts ----------
function renderChart() {{
  const tk = document.getElementById('chart_t').value, t = T[tk];
  const box = document.getElementById('chart_box');
  if (typeof Plotly === 'undefined') {{ box.innerHTML = '<p class="note" style="padding:16px">No se pudo cargar la libreria de graficos (Plotly). Recarga la pagina; si persiste, avisa.</p>'; return; }}
  if (!t || !t.fit_d) {{ box.innerHTML = '<p class="note">Sin ajuste para esta accion.</p>'; return; }}
  const iD = t.fit_d.indexOf(D);
  const hd = t.fit_d.slice(0, iD+1), fd = t.fit_d.slice(iD+1);
  const yh = t.yhat_s.slice(0, iD+1), yf = t.yhat_s.slice(iD+1);
  const lo = t.lo, hi = t.hi;
  const fg = cssVar('--fg'), grid = cssVar('--line'), muted = cssVar('--muted');
  const traces = [
    {{x: hd.concat(hd.slice().reverse()), y: hi.slice(0,iD+1).concat(lo.slice(0,iD+1).reverse()), fill:'toself', fillcolor:'rgba(79,172,222,0.10)', line:{{width:0}}, hoverinfo:'skip', showlegend:false, type:'scatter'}},
    {{x: fd.concat(fd.slice().reverse()), y: hi.slice(iD+1).concat(lo.slice(iD+1).reverse()), fill:'toself', fillcolor:'rgba(79,172,222,0.18)', line:{{width:0}}, hoverinfo:'skip', name:'IC 95 %', type:'scatter'}},
    {{x: fd, y: yf.map(v => v*(1-S.entry/100)), line:{{color:'#27ae60', dash:'dash', width:1.2}}, name:`Umbral compra −${{S.entry}} %`, hoverinfo:'skip'}},
    {{x: fd, y: yf.map(v => v*(1+S.exit/100)), line:{{color:'#e74c3c', dash:'dash', width:1.2}}, name:`Umbral venta +${{S.exit}} %`, hoverinfo:'skip'}},
    {{x: fd, y: yf, line:{{color:'#4facde', width:2}}, name:'Prophet yhat (proyeccion)'}},
    {{x: hd, y: yh, line:{{color:'#4facde', width:1.5, dash:'dot'}}, name:'Prophet yhat (ajustado)'}},
    {{x: t.hist_d, y: t.hist_y, line:{{color:'#4b6584', width:1.5}}, name:'Precio historico'}},
  ];
  if (t.close != null && t.yhat != null) {{
    const dev = (t.close/t.yhat-1)*100, c = dev <= -S.entry ? '#27ae60' : dev >= S.exit ? '#e74c3c' : '#f0a500';
    traces.push({{x:[D], y:[t.close], mode:'markers+text', marker:{{size:11, color:c, line:{{width:2, color:'white'}}}}, text:[`Hoy ${{fmt(t.close)}} $ (${{pct(dev)}})`], textposition:'top center', textfont:{{size:10, color:c}}, name:`Precio hoy ${{pct(dev)}} yhat`}});
  }}
  Plotly.newPlot(box, traces, {{
    title: {{text: `<b>${{lbl(tk)}}</b> — R²=${{fmt(t.r2,2)}}  σ=${{fmt(t.sigma*100,1)}} %  g=${{fmt(t.growth*100,1)}} %/año  ADR=${{fmt(t.adr*100,1)}} %${{passes(t, SC) ? '' : '  (no pasa el screening)'}}`, font:{{size:13, color:fg}}}},
    height: 480, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{{color:fg}},
    legend:{{orientation:'h', y:1.02, yanchor:'bottom', x:0, font:{{size:10}}, bgcolor:'rgba(0,0,0,0)'}},
    xaxis:{{gridcolor:grid, showspikes:true}}, yaxis:{{gridcolor:grid, tickprefix:'$'}}, margin:{{l:50, r:20, t:70, b:40}}, hovermode:'x unified',
    shapes:[{{type:'line', x0:D, x1:D, y0:0, y1:1, yref:'paper', line:{{dash:'dash', color:'rgba(79,172,222,0.5)'}}}}],
  }}, {{displayModeBar:false, responsive:true}});
  const dev = (t.close/t.yhat-1)*100;
  document.getElementById('chart_stats').innerHTML = `<div class="kpis">
    <div class="kpi"><div class="l">Precio actual</div><div class="v">${{usd(t.close)}}</div></div>
    <div class="kpi"><div class="l">yhat hoy</div><div class="v">${{usd(t.yhat)}}</div></div>
    <div class="kpi"><div class="l">Desviacion</div><div class="v ${{dev<0?'poor':'good'}}">${{pct(dev)}}</div></div>
    <div class="kpi"><div class="l">Umbral compra</div><div class="v">${{usd(t.yhat*(1-S.entry/100))}}</div></div>
    <div class="kpi"><div class="l">Umbral venta</div><div class="v">${{usd(t.yhat*(1+S.exit/100))}}</div></div></div>`;
}}
// ---------- Proyecciones ----------
function renderProj() {{
  const sel = document.getElementById('proj_t2').value;
  const days = parseInt(document.getElementById('proj_days').value, 10);
  const tks = sel === '(todas)' ? Object.keys(T).filter(tk => passes(T[tk], SC)).sort() : [sel];
  const rows = [];
  for (const tk of tks) {{
    const t = T[tk]; if (!t.fit_d) continue;
    const iD = t.fit_d.indexOf(D);
    for (let i = iD+1; i < t.fit_d.length; i++) {{
      const d = t.fit_d[i], dd = Math.round((new Date(d) - new Date(D)) / 86400000);
      if (dd > days) break;
      const y = t.yhat_s[i], dev = t.close ? (t.close/y-1)*100 : null;
      const [sig, cls] = dev==null ? ['—','neu'] : classify(dev, S);
      rows.push({{dev: dev==null?0:dev, cells: [lblHtml(tk), d, dd, usd(y), usd(t.lo[i]), usd(t.hi[i]), usd(y*(1-S.entry/100)), usd(y*(1+S.exit/100)), usd(t.close), [pct(dev), dev<0?'poor':'good'], `<span class="pill ${{cls}}">${{sig}}</span>`]}});
    }}
  }}
  const sortBy = document.getElementById('proj_sort').value;
  if (sortBy === 'dev') rows.sort((a,b)=>a.dev-b.dev); else if (sortBy === 'fecha') rows.sort((a,b)=>a.cells[1].localeCompare(b.cells[1])||a.cells[0].localeCompare(b.cells[0]));
  // Resumen: en cuantos dias del horizonte el precio de HOY estaria en zona de compra frente al yhat de ese dia,
  // repartido en primeros / medios / finales dias del horizonte (tercios)
  const allOk = Object.keys(T).filter(tk => passes(T[tk], SC)).sort();
  const summ = [];
  for (const tk of allOk) {{
    const t = T[tk]; if (!t.fit_d || !t.close) continue;
    const iD = t.fit_d.indexOf(D);
    let n = 0, b = [0, 0, 0], tot = 0, firstBuy = null;
    for (let i = iD+1; i < t.fit_d.length; i++) {{
      const dd = Math.round((new Date(t.fit_d[i]) - new Date(D)) / 86400000);
      if (dd > days) break;
      tot++;
      const dev = (t.close / t.yhat_s[i] - 1) * 100;
      if (dev <= -S.entry && dev > -S.max_dev) {{ n++; b[Math.min(2, Math.floor((dd - 1) * 3 / days))]++; if (firstBuy === null) firstBuy = dd; }}
    }}
    if (n > 0) {{
      const devToday = (t.close / t.yhat - 1) * 100;
      summ.push({{n, cells: [lblHtml(tk), usd(t.close), usd(t.yhat), [pct(devToday), devToday<0?'poor':'good'], `${{n}} de ${{tot}}`, b[0], b[1], b[2], firstBuy, `<span class="pill ${{classify(devToday, S)[1]}}">${{classify(devToday, S)[0]}}</span>`]}});
    }}
  }}
  summ.sort((a, b) => b.n - a.n);
  const third = Math.max(1, Math.round(days / 3));
  const summHtml = summ.length ? table('proj_summ', [['Ticker',0],['Precio hoy',1],['yhat hoy',1],['Dev % hoy',1],['Dias en zona de compra',1],[`Primeros (1-${{third}} d)`,1],[`Medios (${{third+1}}-${{2*third}} d)`,1],[`Finales (${{2*third+1}}-${{days}} d)`,1],['Primer dia',1],['Señal hoy',0]], summ.slice(0, 25).map(r => r.cells))
    : '<p class="note">Ninguna accion entra en zona de compra en el horizonte con los umbrales actuales.</p>';
  document.getElementById('proj_out').innerHTML =
    `<h3>Resumen · acciones con mas dias en zona de compra (precio de hoy frente al yhat de cada dia futuro, ${{days}} dias)</h3>` + summHtml +
    `<p class="note">Lectura: si una accion esta en zona de compra solo en los dias "finales", es que el yhat sube con el tiempo y el precio de hoy se queda rezagado; si lo esta en los "primeros" y deja de estarlo, el yhat baja. La columna "Primer dia" es el primer dia del horizonte en que entraria en zona de compra.</p>` +
    `<h3>Detalle por dia</h3><p class="note">${{rows.length}} filas · ${{tks.length}} acciones · Dev % = como estaria el precio de hoy frente al yhat de ese dia futuro.</p>` +
    table('proj_tbl', [['Ticker',0],['Fecha',0],['Dias',1],['yhat',1],['yhat lo',1],['yhat hi',1],['Thr compra',1],['Thr venta',1],['Precio hoy',1],['Dev %',1],['Señal',0]], rows.map(r=>r.cells));
}}
// ---------- Posiciones ----------
function renderPositions() {{
  const open = DATA.positions.filter(p => !p.cerrada), closed = DATA.positions.filter(p => p.cerrada);
  let h = '';
  if (!open.length) h += '<p class="note">No hay posiciones abiertas registradas en la app.</p>';
  else {{
    const rows = open.map(p => {{
      const t = T[p.ticker] || {{}}, cpx = t.close, epx = p.precio_entrada, yh = t.yhat;
      const pnl = cpx ? (cpx/epx-1)*100 : null, days = Math.round((new Date(D)-new Date(p.fecha_entrada))/86400000);
      const slp = epx*(1-S.sl/100), tpp = yh ? yh*(1+S.exit/100) : null;
      const est = days >= S.hold ? '⏰ VENCER PRONTO' : (cpx && cpx <= slp) ? '🛑 SL ACTIVADO' : (tpp && cpx && cpx >= tpp) ? '✅ TP ALCANZADO' : '⏳ Activa';
      return [lblHtml(p.ticker), esc(p.plataforma||'—'), p.cantidad!=null ? fmt(p.cantidad, 4) : '—', p.fecha_entrada, usd(epx), p.precio_entrada_eur ? fmt(p.precio_entrada_eur)+' €' : '—', fmt(p.capital)+' €', usd(cpx), [pct(pnl), pnl>0?'good':pnl<0?'poor':''], pnl==null?'—':fmt(p.capital*pnl/100)+' €', days, usd(slp), usd(tpp), est, esc(p.nota||'')];
    }});
    h += table('pos_tbl', [['Ticker',0],['Plataforma',0],['Cantidad',1],['Entrada',0],['Prec. ent. $',1],['Prec. ent. €',1],['Capital €',1],['Prec. hoy $',1],['P&L %',1],['P&L €',1],['Dias',1],['SL ref',1],['TP ref (yhat)',1],['Estado',0],['Nota',0]], rows);
  }}
  h += '<h3>Mis salidas</h3>';
  if (!closed.length) h += '<p class="note">Sin ventas registradas.</p>';
  else h += table('cl_tbl', [['Ticker',0],['Plataforma',0],['Cantidad',1],['Entrada',0],['Prec. ent. $',1],['Salida',0],['Prec. sal. $',1],['Dias',1],['Capital €',1],['P&L %',1],['P&L €',1],['Motivo',0]],
    closed.map(p => [lblHtml(p.ticker), esc(p.plataforma||'—'), p.cantidad!=null ? fmt(p.cantidad, 4) : '—', p.fecha_entrada, usd(p.precio_entrada), p.fecha_salida, usd(p.precio_salida), Math.round((new Date(p.fecha_salida)-new Date(p.fecha_entrada))/86400000), fmt(p.capital,0)+' €', [pct(p.pnl_pct), p.pnl_pct>0?'good':'poor'], [fmt(p.pnl_eur)+' €', p.pnl_eur>0?'good':'poor'], esc(p.motivo_salida||'')]));
  h += '<p class="note">Las entradas y salidas se registran desde la app de Streamlit; aqui se muestran en solo lectura.</p>';
  const plog = DATA.positions_log || [];
  h += `<h3>Diario de decisiones de mis posiciones</h3>` + (plog.length ? `<details open><summary>${{plog.length}} sesiones (la mas reciente arriba)</summary>` +
    plog.slice().reverse().map(e => `<div class="day"><b>${{e.date}}</b><ul>${{e.events.map(x=>`<li>${{esc(x)}}</li>`).join('')}}</ul></div>`).join('') + '</details>'
    : '<p class="note">Todavia no hay sesiones registradas.</p>');
  document.getElementById('pos_out').innerHTML = h;
}}
// ---------- Carteras de papel ----------
function spark(points, base=1000) {{
  if (!points || points.length < 2) return '';
  const ys = points.map(p=>p.equity), w=320, h=70, lo=Math.min(...ys, base), hi0=Math.max(...ys, base), hi = hi0===lo ? lo+1 : hi0;
  const X = i => 4 + i*(w-8)/(ys.length-1), Y = v => 4 + (hi-v)*(h-8)/(hi-lo);
  const path = ys.map((v,i)=>`${{i?'L':'M'}}${{X(i).toFixed(1)}},${{Y(v).toFixed(1)}}`).join(' ');
  const c = ys[ys.length-1] >= base ? 'var(--ok)' : 'var(--bad)';
  return `<svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="${{h}}"><path d="${{path}} L${{X(ys.length-1).toFixed(1)}},${{h-4}} L4,${{h-4}} Z" fill="${{c}}" opacity="0.12"/><line x1="4" x2="${{w-4}}" y1="${{Y(base).toFixed(1)}}" y2="${{Y(base).toFixed(1)}}" stroke="var(--muted)" stroke-dasharray="3 3"/><path d="${{path}}" fill="none" stroke="${{c}}" stroke-width="2"/></svg>`;
}}
function renderPaper(key) {{
  const st = DATA.paper[key], cfg = DATA.cfgs[key];
  const closed = st.closed, n = closed.length, wins = closed.filter(c=>c.pnl_cash>0).length;
  const eq = st.equity.length ? st.equity[st.equity.length-1].equity : 1000, ret = (eq/1000-1)*100;
  const pnlC = closed.reduce((a,c)=>a+c.pnl_cash,0);
  const rows = st.slots.map((slot,i) => {{
    const p = slot.position, o = slot.pending;
    if (p) {{ const t = T[p.ticker]||{{}}, cpx = t.close, pnl = cpx ? (cpx/p.fill_px-1)*100 : null, yh = t.yhat;
      return [i+1, 'abierta' + (o && o.side==='sell' ? ` · venta pendiente (${{o.reason}})` : ''), lblHtml(p.ticker), p.shares, p.signal_date, pct(p.signal_dev), pct(p.signal_drop), p.fill_date, usd(p.fill_px), usd(p.cash_in), usd(cpx), [pct(pnl), pnl>0?'good':pnl<0?'poor':''], [usd(cpx ? (cpx-p.fill_px)*p.shares : null), pnl>0?'good':pnl<0?'poor':''], Math.round((new Date(D)-new Date(p.fill_date))/86400000), usd(yh ? yh*(1+cfg.exit_dev/100) : null), usd(p.fill_px*(1-cfg.sl/100))]; }}
    if (o) return [i+1, 'compra pendiente (apertura siguiente)', lblHtml(o.ticker), '—', o.signal_date, pct(o.dev), pct(o.drop), '—','—','—', usd((T[o.ticker]||{{}}).close), '—','—','—','—','—'];
    return [i+1, 'libre', '—','—','—','—','—','—','—','—','—','—','—','—','—','—'];
  }});
  const crow = closed.slice().sort((a,b)=>b.exit_date.localeCompare(a.exit_date)).map(c => [lblHtml(c.ticker), c.shares||'—', c.signal_date, pct(c.signal_dev), c.fill_date, usd(c.fill_px), c.exit_date, usd(c.exit_px), c.exit_reason, c.days, [pct(c.pnl_pct*100), c.pnl_pct>0?'good':'poor'], [usd(c.pnl_cash), c.pnl_cash>0?'good':'poor']]);
  const log = st.log.slice().reverse().map(e => `<div class="day"><b>${{e.date}}</b>${{e.oos ? ' <span class="pill buy">fuera de muestra</span>' : ''}}<ul>${{e.events.map(x=>`<li>${{esc(x)}}</li>`).join('')}}</ul></div>`).join('');
  const scr = (st.screening||[]).map(p => `desde ${{p.from}}: ${{esc(p.name)}}`).join(' · ');
  document.getElementById('paper_'+key).innerHTML = `
    <p class="note">Entrada: precio entre ${{cfg.entry_dev}} % y ${{cfg.max_dev}} % bajo yhat${{cfg.drop ? ` <b>y</b> caida del dia ≥ ${{cfg.drop}} %` : ''}}, con screening. Salida (en este orden): ${{cfg.hold}} dias → stop loss ${{cfg.sl}} % → precio ≥ yhat +${{cfg.exit_dev}} %. 1.000 $ en una bolsa comun, hasta 3 posiciones, <b>acciones enteras</b> (objetivo por compra = efectivo / huecos libres, redondeado al entero mas cercano), 1 $ por operacion. Cada decision usa solo los datos de ese dia y se ejecuta a la apertura siguiente.</p>
    <p class="note"><b>Screening congelado por fechas</b> (no cambia aunque muevas los sliders): ${{scr}}. <b>Fuera de muestra desde ${{st.oos_from||'—'}}</b>: antes de esa fecha el perfil de screening se habia elegido con datos que incluian la ventana simulada (hallazgo de la auditoria del 4-nov), por lo que sep-nov se regenero con el perfil previo a la ventana.</p>
    <div class="kpis"><div class="kpi"><div class="l">Valor cartera</div><div class="v ${{ret>0?'good':ret<0?'poor':''}}">${{usd(eq)}}</div><div class="n">${{pct(ret)}} desde el inicio</div></div>
      <div class="kpi"><div class="l">Efectivo</div><div class="v">${{usd(st.cash)}}</div></div>
      <div class="kpi"><div class="l">Ultima sesion</div><div class="v" style="font-size:16px">${{st.last_date||'—'}}</div></div>
      <div class="kpi"><div class="l">Cerradas</div><div class="v">${{n}}</div><div class="n">acierto ${{n?Math.round(100*wins/n)+' %':'—'}}</div></div>
      <div class="kpi"><div class="l">P&amp;L cerrado</div><div class="v ${{pnlC>0?'good':pnlC<0?'poor':''}}">${{usd(pnlC)}}</div></div>
      <div class="kpi spark"><div class="l">Evolucion</div>${{spark(st.equity)}}</div></div>
    <h3>Posiciones y ordenes</h3>${{table('pp_'+key, [['Slot',1],['Estado',0],['Ticker',0],['Acciones',1],['Señal',0],['Bajo yhat',1],['Caida dia',1],['Compra',0],['Precio compra',1],['Invertido',1],['Precio hoy',1],['P&L',1],['P&L $',1],['Dias',1],['Vender si ≥',1],['Stop loss',1]], rows)}}
    <h3>Operaciones cerradas</h3>${{crow.length ? table('pc_'+key, [['Ticker',0],['Acciones',1],['Señal',0],['Bajo yhat',1],['Compra',0],['Precio',1],['Venta',0],['Precio',1],['Motivo',0],['Dias',1],['P&L',1],['P&L $',1]], crow) : '<p class="note">Todavia no hay operaciones cerradas.</p>'}}
    <details><summary>Diario de decisiones, dia a dia desde el inicio (${{st.log.length}} sesiones; la mas reciente arriba)</summary>${{log}}</details>`;
}}
// ---------- Montaje ----------
const TABS = [
  ['hoy', '🚨 Señales de hoy', () => presetRow('') + `<div class="controls">${{slider('s_entry','Entrada: caida bajo yhat',1,12,1,S.entry,' %')}}${{slider('s_exit','Salida: subida sobre yhat',3,30,1,S.exit,' %')}}${{slider('s_sl','Stop loss',3,20,1,S.sl,' %')}}${{slider('s_hold','Dias max en posicion',5,90,1,S.hold,' d')}}${{slider('s_maxdev','Cap caida maxima',5,30,1,S.max_dev,' %')}}</div><div id="sig_out"></div>`],
  ['scr', '🔍 Screening', () => presetRow('_s') + `<div class="controls">${{slider('c_r2','R² minimo',0.5,0.99,0.01,SC.min_r2)}}${{slider('c_sig','Sigma max',2,15,1,SC.max_sigma,' %')}}${{slider('c_gmin','Crecimiento min',-50,30,1,SC.min_growth,' %')}}${{slider('c_gmax','Crecimiento max',5,120,1,SC.max_growth,' %')}}${{slider('c_adr','ADR max',1,10,1,SC.max_adr,' %')}}</div><div id="scr_out"></div>`],
  ['chart', '📊 Charts', () => `<div class="controls"><div class="ctl"><label for="chart_q">Buscar (ticker o nombre)</label><input type="search" id="chart_q" list="chart_list" placeholder="p. ej. MRVL o Marvell" autocomplete="off" style="width:100%;font:13px var(--sans);color:var(--fg);background:var(--panel);border:1px solid var(--line);padding:5px 8px"><datalist id="chart_list"></datalist></div><div class="ctl"><label for="chart_t">Accion</label><select id="chart_t"></select></div></div><div class="chart" id="chart_box"></div><div id="chart_stats"></div>`],
  ['proj', '📋 Proyecciones', () => `<div class="controls"><div class="ctl"><label>Acciones</label><select id="proj_t2"></select></div><div class="ctl"><label>Horizonte (dias)</label><select id="proj_days"><option>7</option><option>14</option><option>21</option><option selected>28</option><option>35</option><option>42</option></select></div><div class="ctl"><label>Ordenar por</label><select id="proj_sort"><option value="dev">Dev % (menor primero)</option><option value="fecha">Fecha</option><option value="ticker">Ticker</option></select></div></div><div id="proj_out"></div>`],
  ['pos', '💼 Mis posiciones', () => `<div id="pos_out"></div>`],
  ['bt15', '🧪 Claude · Backtest 15', () => `<div id="paper_bt15"></div>`],
  ['combo', '🧪 Claude · Combinado', () => `<div id="paper_combo"></div>`],
  ['docs', '📚 Como funciona Prophet', () => `<div class="docs">${{DOCS_HTML}}</div>`],
  ['ver', 'ℹ️ Versiones', () => VERSIONS_HTML],
];
const tabsEl = document.getElementById('tabs'), panelsEl = document.getElementById('panels');
TABS.forEach(([id, label, render]) => {{
  const b = document.createElement('button'); b.className = 'tab'; b.role = 'tab'; b.id = 'tab_'+id; b.textContent = label; b.setAttribute('aria-selected', 'false');
  b.onclick = () => show(id); tabsEl.appendChild(b);
  const p = document.createElement('div'); p.className = 'panel'; p.id = 'panel_'+id; p.hidden = true; p.innerHTML = render(); panelsEl.appendChild(p);
}});
function show(id) {{
  TABS.forEach(([k]) => {{ document.getElementById('panel_'+k).hidden = k !== id; document.getElementById('tab_'+k).setAttribute('aria-selected', String(k === id)); }});
  try {{ location.hash = id; }} catch (e) {{}}
  if (id === 'chart') renderChart();
}}
bindSliders([['s_entry','entry',' %'],['s_exit','exit',' %'],['s_sl','sl',' %'],['s_hold','hold',' d'],['s_maxdev','max_dev',' %']], S, () => {{ PRESET = 'manual'; DROP_REQ = null; renderSignals(); renderProj(); renderPositions(); if (!document.getElementById('panel_chart').hidden) renderChart(); }});
bindSliders([['c_r2','min_r2',''],['c_sig','max_sigma',' %'],['c_gmin','min_growth',' %'],['c_gmax','max_growth',' %'],['c_adr','max_adr',' %']], SC, () => {{ PRESET = 'manual'; DROP_REQ = null; renderScreening(); renderSignals(); renderProj(); }});
bindUserButtons();
initPresetsDb();
document.getElementById('chart_t').onchange = renderChart;
function findTicker(q) {{
  q = (q || '').trim().toUpperCase(); if (!q) return null;
  const all = Object.keys(T).filter(tk => T[tk].fit_d);
  const tkPart = q.split('·')[0].trim();
  return all.find(tk => tk === tkPart) || all.find(tk => tk.startsWith(tkPart)) ||
         all.find(tk => (NM[tk]||'').toUpperCase().includes(q)) || null;
}}
const chartQ = document.getElementById('chart_q');
chartQ.oninput = () => {{ const tk = findTicker(chartQ.value); if (tk) {{ document.getElementById('chart_t').value = tk; renderChart(); }} }};
chartQ.onkeydown = e => {{ if (e.key === 'Enter') {{ e.preventDefault(); chartQ.oninput(); }} }};
['proj_t2','proj_days','proj_sort'].forEach(id => document.getElementById(id).onchange = renderProj);
renderScreening(); renderSignals(); renderProj(); renderPositions(); renderPaper('bt15'); renderPaper('combo');
const start = (location.hash||'').replace('#','');
show(TABS.some(t=>t[0]===start) ? start : 'hoy');
window.addEventListener('hashchange', () => {{ const h = location.hash.replace('#',''); if (TABS.some(t=>t[0]===h)) show(h); }});
</script>
"""
