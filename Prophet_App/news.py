"""Ultimas noticias de las acciones candidatas y clasificacion de la caida: mercado/sector vs noticia propia.

Fuente: feed RSS de Yahoo Finance por ticker (sin clave). Si falla, Google News RSS.
La clasificacion es una ayuda, no una verdad: combina
  1) la caida del dia frente a la del Nasdaq (QQQ) ese mismo dia (dato objetivo, sin mirar el futuro), y
  2) palabras clave en los titulares de los ultimos dias (que tipo de noticia domina).
"""
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

UA = {"User-Agent": "Mozilla/5.0 (prophet-trader news)"}
DAYS = 7          # ventana de titulares
MAX_ITEMS = 8

COMPANY_RE = re.compile(
    r"\b(earnings|results|quarter(ly)?|q[1-4]|guidance|outlook|forecasts?|revenue|sales|profits?|margins?|miss(es|ed)?|"
    r"beats?|downgrades?|upgrades?|price target|cuts?|slash(es|ed)?|lawsuits?|probe|investigation|sec|recall|ceo|cfo|"
    r"resign(s|ed)?|layoffs?|orders?|customers?|contracts?|acquisition|acquire[sd]?|merger|fda|deal|dividend|buyback)\b", re.I)
MARKET_RE = re.compile(
    r"\b(sell-?off|wall street|nasdaq|s&p|dow( jones)?|fed|federal reserve|interest rates?|inflation|elections?|midterms?|"
    r"tariffs?|recession|treasury|yields?|macro|(chip|ai|tech|travel|semiconductor|bank|retail|trucking|freight|software|energy) stocks|"
    r"sector|market(s)?|stock market|index)\b", re.I)


def tag_title(title):
    c, m = bool(COMPANY_RE.search(title)), bool(MARKET_RE.search(title))
    return "ambas" if c and m else "empresa" if c else "mercado" if m else ""


def _parse(xml_bytes, source_default=""):
    out = []
    for it in ET.fromstring(xml_bytes).findall(".//item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:
            dt = None
        src = (it.findtext("source") or source_default).strip()
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        out.append(dict(date=(dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M") if dt else ""), dt=dt,
                        title=title, link=(it.findtext("link") or "").strip(), source=src, tag=tag_title(title)))
    return out


def _key_word(ticker, name):
    """Palabra del nombre para reconocer titulares de la empresa (p. ej. 'Nvidia', 'Dominion', 'Zebra')."""
    w = [x for x in re.split(r"[\s(),.]+", name or "") if x]
    if not w:
        return None
    return w[0] if len(w[0]) >= 4 or len(w) == 1 else w[1]


def fetch(ticker, name=""):
    """Titulares de los ultimos DAYS dias. Primero los que mencionan a la empresa (rel=True), luego el resto,
    cada grupo del mas reciente al mas antiguo. Yahoo Finance; si da menos de 3 de la empresa, completa con Google News.
    Devuelve (items, error)."""
    since = datetime.now(timezone.utc) - timedelta(days=DAYS)
    kw = _key_word(ticker, name)
    rel_re = re.compile(r"\b(" + re.escape(ticker) + (r"|" + re.escape(kw) if kw else "") + r")", re.I)
    q = requests.utils.quote(f'"{kw or ticker}" {ticker} stock')
    urls = [(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US", "Yahoo Finance"),
            (f"https://news.google.com/rss/search?q={q}+when:{DAYS}d&hl=en-US&gl=US&ceid=US:en", "")]
    items, seen, err = [], set(), None
    for url, src in urls:
        if sum(1 for x in items if x["rel"]) >= 3:
            break
        try:
            r = requests.get(url, headers=UA, timeout=15)
            r.raise_for_status()
            for x in _parse(r.content, src):
                k = x["title"].lower()[:70]
                if (x["dt"] is None or x["dt"] >= since) and k not in seen:
                    seen.add(k)
                    x["rel"] = bool(rel_re.search(x["title"]))
                    items.append(x)
        except Exception as e:
            err = str(e)[:80]
    items.sort(key=lambda x: (not x["rel"], -(x["dt"] or since).timestamp()))
    for x in items:
        x.pop("dt", None)
    return items[:MAX_ITEMS], (None if items else err or "sin titulares en los ultimos dias")


def classify(drop, qqq, items):
    """drop y qqq en % del dia. Devuelve (codigo, etiqueta, explicacion)."""
    use = [x for x in items if x.get("rel")] or items     # cuentan solo los titulares que mencionan a la empresa
    n_c = sum(1 for x in use if x["tag"] in ("empresa", "ambas"))
    n_m = sum(1 for x in use if x["tag"] in ("mercado", "ambas"))
    kw = f"titulares sobre la empresa: {n_c} de tipo empresa, {n_m} de mercado/sector"
    if drop is None:
        return "na", "sin dato", kw
    rel = drop - (qqq or 0)
    base = f"hoy {drop:+.1f} % vs Nasdaq {qqq:+.1f} % (diferencia {rel:+.1f} pts); {kw}"
    if drop > -3:
        return "flat", "sin caida fuerte hoy", base + ". La caida es acumulada de dias anteriores: mira los titulares de la semana."
    if qqq is not None and qqq <= -1.5 and rel > -4:
        return "mkt", "🌊 mercado", base + ". Cayo todo el mercado y esta accion no mucho mas: parece panico general (en los 6 casos revisados este tipo tendio a recuperarse)."
    if rel <= -4 and n_c > n_m:
        return "co", "🏢 noticia propia", base + ". Cayo mucho mas que el mercado y dominan noticias de la empresa (resultados, guia, clientes...): en los casos revisados este tipo siguio bajando."
    if rel <= -4 and n_m >= n_c and n_m > 0:
        return "sec", "🌊 sector", base + ". Cayo mas que el Nasdaq pero los titulares hablan del sector o del mercado: posible miedo sectorial (como ABNB en sep). Revisa los titulares."
    return "mix", "❓ revisar", base + ". No esta claro: lee los titulares antes de decidir."
