"""Seleccion walk-forward y veredicto.

select(): con level1_summary (prophet, post_close, screening ON):
  - para cada periodo objetivo Y (P3, P4, P5, HOLD) elige la combo con mejor rango medio de alpha
    en los periodos anteriores a Y (consistencia, no maximo), exige n>=30 por periodo de entreno,
    y comprueba que sus vecinas en el grid tambien estan arriba (meseta). Reporta el resultado en Y.
  - LOYO sobre los 5 periodos completos.
  - escribe shortlist_refit{N}.json para portfolio.
report(): tablas + reglas de decision pre-registradas.
"""
import os, json, time
import numpy as np
import pandas as pd

from . import common as C
from .level1 import ENTRY, MAXDEV, SL, TP, HOLD, KEY_CONFIGS

FULL = [p[0] for p in C.PERIODS[:5]]
OOS_TARGETS = ["P3_2023-24", "P4_2024-25", "P5_2025-26", "HOLD_2026"]
MIN_N = 30


def _main(summ):
    return summ[(summ["signal"] == "prophet") & (summ["protocol"] == "post_close") & (summ["screen"] == "True")].copy()


def _neighbors(row):
    """Combos a un paso en cada eje (misma familia de salida)."""
    out = []
    for axis, grid in [("entry", ENTRY), ("max_dev", MAXDEV), ("sl", SL), ("tp", TP), ("hold", HOLD)]:
        v = row[axis]
        if axis in ("sl", "tp") and (row["devx"] is not None and not pd.isna(row["devx"])):
            continue
        lst = list(grid)
        vv = None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
        if vv not in lst:
            continue
        i = lst.index(vv)
        for j in (i - 1, i + 1):
            if 0 <= j < len(lst):
                k = dict(entry=row["entry"], max_dev=row["max_dev"], sl=row["sl"], tp=row["tp"], devx=row["devx"], hold=row["hold"])
                k[axis] = lst[j]
                out.append(k)
    return out


def _key(k):
    def f(v):
        return "None" if (v is None or (isinstance(v, float) and np.isnan(v))) else (int(v) if float(v).is_integer() else v)
    return f"e{f(k['entry'])}_md{f(k['max_dev'])}_sl{f(k['sl'])}_tp{f(k['tp'])}_dx{f(k['devx'])}_h{f(k['hold'])}"


def _rank_table(m, periods):
    """Score por combo = rango medio de mean_alpha en los periodos dados (peor rango si n<MIN_N)."""
    sub = m[m["period"].isin(periods)]
    piv = sub.pivot_table(index="key", columns="period", values="mean_alpha")
    npiv = sub.pivot_table(index="key", columns="period", values="n_closed")
    piv = piv.where(npiv >= MIN_N)
    ranks = piv.rank(ascending=False)
    worst = ranks.max().max()
    ranks = ranks.fillna(worst)
    score = ranks.mean(axis=1)
    return score.sort_values(), piv


def select(refit_every=5):
    summ = pd.read_parquet(os.path.join(C.RES_DIR, f"level1_summary_refit{refit_every}.parquet"))
    m = _main(summ)
    meta = m.drop_duplicates("key").set_index("key")[["entry", "max_dev", "sl", "tp", "devx", "hold"]]
    chain, shortlist = [], {}
    for Y in OOS_TARGETS:
        train = [p for p in FULL if FULL.index(p) < (FULL.index(Y) if Y in FULL else 5)]
        if len(train) < 2:
            continue
        score, piv = _rank_table(m, train)
        best = None
        for key in score.index[:50]:
            row = meta.loc[key]
            nb = [_key(k) for k in _neighbors(row)]
            nb = [k for k in nb if k in score.index]
            if not nb:
                continue
            nb_score = score.loc[nb].mean()
            if nb_score <= score.quantile(0.40):
                best = key
                break
        best = best or score.index[0]
        res = m[(m["key"] == best) & (m["period"] == Y)]
        c0 = m[(m["key"] == _key(KEY_CONFIGS["C0"])) & (m["period"] == Y)]
        chain.append(dict(target=Y, trained_on=",".join(train), chosen=best, score=float(score.loc[best]),
                          c0_score=float(score.get(_key(KEY_CONFIGS["C0"]), np.nan)),
                          c0_rank_pct=float((score < score.get(_key(KEY_CONFIGS["C0"]), np.inf)).mean()),
                          oos_alpha=float(res["mean_alpha"].iloc[0]) if len(res) else np.nan,
                          oos_net=float(res["mean_net"].iloc[0]) if len(res) else np.nan,
                          oos_n=int(res["n_closed"].iloc[0]) if len(res) else 0,
                          c0_alpha=float(c0["mean_alpha"].iloc[0]) if len(c0) else np.nan,
                          c0_net=float(c0["mean_net"].iloc[0]) if len(c0) else np.nan))
        shortlist[f"WF_{Y}"] = {k: (None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v))) for k, v in meta.loc[best].items()}
    # LOYO sobre los 5 periodos completos
    loyo = []
    for Y in FULL:
        train = [p for p in FULL if p != Y]
        score, _ = _rank_table(m, train)
        best = score.index[0]
        res = m[(m["key"] == best) & (m["period"] == Y)]
        loyo.append(dict(held_out=Y, chosen=best, oos_alpha=float(res["mean_alpha"].iloc[0]) if len(res) else np.nan,
                         oos_n=int(res["n_closed"].iloc[0]) if len(res) else 0))
    # top-5 consistencia sobre los 5 periodos (referencia en muestra)
    score_all, piv_all = _rank_table(m, FULL)
    for i, key in enumerate(score_all.index[:5]):
        shortlist[f"TOP{i+1}_insample"] = {k: (None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v))) for k, v in meta.loc[key].items()}
    # mejor control sin Prophet (misma regla C0 con linreg/sma) para referencia
    chain_df, loyo_df = pd.DataFrame(chain), pd.DataFrame(loyo)
    chain_df.to_parquet(os.path.join(C.RES_DIR, f"select_chain_refit{refit_every}.parquet"), index=False)
    loyo_df.to_parquet(os.path.join(C.RES_DIR, f"select_loyo_refit{refit_every}.parquet"), index=False)
    score_all.rename("score").reset_index().to_parquet(os.path.join(C.RES_DIR, f"select_scores_refit{refit_every}.parquet"), index=False)
    with open(os.path.join(C.RES_DIR, f"shortlist_refit{refit_every}.json"), "w") as f:
        json.dump(shortlist, f, indent=2)
    print("[select] cadena walk-forward:"); print(chain_df.round(4).to_string())
    print("[select] LOYO:"); print(loyo_df.round(4).to_string())
    return chain_df, loyo_df, score_all


def _fmt(x, pct=True, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x*100:+.{d}f}%" if pct else f"{x:.{d}f}"


def run(refit_every=5):
    R = C.RES_DIR
    summ = pd.read_parquet(os.path.join(R, f"level1_summary_refit{refit_every}.parquet"))
    m = _main(summ)
    c0k = _key(KEY_CONFIGS["C0"])
    lines = [f"# Backtest 15 - walk-forward honesto (refit cada {refit_every} sesiones)", ""]
    lines += [f"Generado {pd.Timestamp.now():%Y-%m-%d %H:%M}. Universo 87 tickers. Periodos ago->jul; P5 es EN MUESTRA para la config actual y la lista de 55.", ""]

    # 1. Config actual por periodo y protocolo
    lines += ["## 1. Config actual C0 (entrada 3 / max_dev 15 / SL 15 / TP 20 / hold 15), nivel 1", "",
              "| periodo | protocolo | n | neto/op | alpha vs incondicional | vs QQQ | win | %hold | %SL | %TP | dias |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for proto in ["post_close", "pre_open"]:
        sub = summ[(summ["key"] == c0k) & (summ["signal"] == "prophet") & (summ["protocol"] == proto) & (summ["screen"] == "True")]
        for _, r in sub.sort_values("period").iterrows():
            lines.append(f"| {r['period']} | {proto} | {int(r['n_closed'])} | {_fmt(r['mean_net'],d=2)} | {_fmt(r['mean_alpha'],d=2)} | {_fmt(r['mean_qqq_excess'],d=2)} | {r['win_rate']*100:.0f}% | {r['sh_hold']*100:.0f}% | {r['sh_sl']*100:.0f}% | {r['sh_tp']*100:.0f}% | {r['mean_days']:.0f} |")
    # variantes de C0: solo k=0 (ajuste del mismo dia), screening en vivo, y la config en vivo C_LIVE
    lines += ["", "### Variantes (post_close): k=0 = solo dias con ajuste del mismo dia (lo que ve la app); 'live' = screening.json actual; C_LIVE = estrategia.json actual", "",
              "| variante | periodo | n | neto/op | alpha | vs QQQ | %hold | %SL | %TP |", "|---|---|---|---|---|---|---|---|---|"]
    clk = _key(KEY_CONFIGS["C_LIVE"])
    for label, key, sname, scr in [("C0 k=0", c0k, "prophet_k0", "True"), ("C0 screening live", c0k, "prophet", "live"),
                                   ("C_LIVE 2/15/SL6/TP7/15", clk, "prophet", "True"), ("C_LIVE screening live", clk, "prophet", "live")]:
        sub = summ[(summ["key"] == key) & (summ["signal"] == sname) & (summ["protocol"] == "post_close") & (summ["screen"] == scr)]
        for _, r in sub.sort_values("period").iterrows():
            lines.append(f"| {label} | {r['period']} | {int(r['n_closed'])} | {_fmt(r['mean_net'],d=2)} | {_fmt(r['mean_alpha'],d=2)} | {_fmt(r['mean_qqq_excess'],d=2)} | {r['sh_hold']*100:.0f}% | {r['sh_sl']*100:.0f}% | {r['sh_tp']*100:.0f}% |")
    # controles sin Prophet
    lines += ["", "### Misma regla con señal sin Prophet (screening OFF para los tres, post_close)", "",
              "| señal | periodo | n | neto/op | alpha |", "|---|---|---|---|---|"]
    for sname in ["prophet", "linreg", "sma100"]:
        sub = summ[(summ["key"] == c0k) & (summ["signal"] == sname) & (summ["protocol"] == "post_close") & (summ["screen"] == "False")]
        for _, r in sub.sort_values("period").iterrows():
            lines.append(f"| {sname} | {r['period']} | {int(r['n_closed'])} | {_fmt(r['mean_net'],d=2)} | {_fmt(r['mean_alpha'],d=2)} |")

    # 2. Seleccion walk-forward
    chain, loyo, score_all = select(refit_every)
    lines += ["", "## 2. Seleccion walk-forward (parametros del año Y elegidos solo con años < Y)", "",
              "| objetivo | entrenado en | elegida | alpha OOS | neto OOS | n | C0 alpha | C0 neto | C0 percentil en entreno |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in chain.iterrows():
        lines.append(f"| {r['target']} | {r['trained_on']} | {r['chosen']} | {_fmt(r['oos_alpha'],d=2)} | {_fmt(r['oos_net'],d=2)} | {r['oos_n']} | {_fmt(r['c0_alpha'],d=2)} | {_fmt(r['c0_net'],d=2)} | {r['c0_rank_pct']*100:.0f}% |")
    lines += ["", "LOYO (5 periodos completos):", "", "| fuera | elegida | alpha OOS | n |", "|---|---|---|---|"]
    for _, r in loyo.iterrows():
        lines.append(f"| {r['held_out']} | {r['chosen']} | {_fmt(r['oos_alpha'],d=2)} | {r['oos_n']} |")
    lines += ["", "Top-10 por consistencia en los 5 periodos (EN MUESTRA, referencia):", "", "| combo | score (rango medio) | " + " | ".join(FULL) + " |", "|---|---|" + "---|" * len(FULL)]
    piv = m.pivot_table(index="key", columns="period", values="mean_alpha")
    for key in score_all.index[:10]:
        lines.append(f"| {key} | {score_all[key]:.0f} | " + " | ".join(_fmt(piv.loc[key].get(p, np.nan), d=2) for p in FULL) + " |")

    # 3. Efectos marginales
    lines += ["", "## 3. Efecto marginal de cada parametro (alpha media por periodo, resto promediado)", ""]
    for axis in ["entry", "max_dev", "sl", "tp", "hold", "devx"]:
        g = m[m["period"].isin(FULL)].assign(v=m[axis].astype(str)).groupby(["v", "period"])["mean_alpha"].mean().unstack()
        lines += [f"**{axis}**", "", "| valor | " + " | ".join(FULL) + " | media |", "|---|" + "---|" * (len(FULL) + 1)]
        for v, r in g.iterrows():
            lines.append(f"| {v} | " + " | ".join(_fmt(r.get(p, np.nan), d=2) for p in FULL) + f" | {_fmt(r.mean(), d=2)} |")
        lines.append("")

    # 4. Curva por bin de desviacion
    db = pd.read_parquet(os.path.join(R, f"level1_devbins_refit{refit_every}.parquet"))
    lines += ["## 4. Alpha a 15 sesiones por bin de desviacion (prophet, screening ON)", "",
              "| bin dev | " + " | ".join(FULL) + " | n total |", "|---|" + "---|" * (len(FULL) + 1)]
    d15 = db[(db["signal"] == "prophet") & (db["h"] == 15)]
    pv = d15.pivot_table(index="bin", columns="period", values="alpha")
    nn = d15.groupby("bin")["n"].sum()
    for b in sorted(pv.index):
        if nn[b] < 50:
            continue
        lines.append(f"| {b}..{b+1}% | " + " | ".join(_fmt(pv.loc[b].get(p, np.nan), d=2) for p in FULL) + f" | {int(nn[b])} |")
    mm = pd.read_parquet(os.path.join(R, f"level1_maemfe_refit{refit_every}.parquet"))
    lines += ["", f"MAE/MFE de la señal base (entrada 3, max_dev 15, hold 15, sin SL/TP), {len(mm)} operaciones: "
              f"MAE p10/p25/p50 = {_fmt(mm['mae'].quantile(.1))} / {_fmt(mm['mae'].quantile(.25))} / {_fmt(mm['mae'].quantile(.5))}; "
              f"MFE p50/p75/p90 = {_fmt(mm['mfe'].quantile(.5))} / {_fmt(mm['mfe'].quantile(.75))} / {_fmt(mm['mfe'].quantile(.9))}. "
              f"Resultado final medio si MAE <= -10%: {_fmt(mm[mm['mae'] <= -0.10]['final'].mean())} (n={int((mm['mae'] <= -0.10).sum())}); "
              f"si MFE >= +10%: {_fmt(mm[mm['mfe'] >= 0.10]['final'].mean())} (n={int((mm['mfe'] >= 0.10).sum())})."]

    # 5. Cartera
    pp = os.path.join(R, f"portfolio_summary_refit{refit_every}.parquet")
    if os.path.exists(pp):
        pf = pd.read_parquet(pp)
        lines += ["", "## 5. Cartera (3 slots, 1.000 EUR por periodo, neto en EUR)", "",
                  "| config | protocolo | universo | periodo | retorno | QQQ | EW 87 | EW 55 | QQQ ajustado exposicion | random | max DD | en mercado | ops | fees |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        pf3 = pf[(pf["slots"] == 3) & (pf["screen"].isin(["True", "live"]))].copy()
        # etiqueta en-muestra / fuera de muestra por fila: WF_<Y> solo es OOS en Y; TOP*_insample nunca;
        # C0/LIST_07 son en-muestra en P5 (ahi se eligieron en los scripts 07-14)
        def _oos(r):
            if r["cfg"].startswith("WF_"):
                return "OOS" if r["cfg"] == f"WF_{r['period']}" else "entreno"
            if r["cfg"].startswith("TOP"):
                return "en muestra"
            if r["cfg"] == "C0" and r["period"] == "P5_2025-26":
                return "en muestra"
            return "OOS"
        pf3["oos"] = pf3.apply(_oos, axis=1)
        pf3 = pf3[pf3["oos"] != "entreno"]
        lines[-2] = "| config | protocolo | universo | periodo | muestra | retorno | QQQ | EW 87 | EW 55 | QQQ ajustado exposicion | random | max DD | en mercado | ops | fees |"
        lines[-1] = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        for _, r in pf3.sort_values(["cfg", "protocol", "universe", "period"]).iterrows():
            lines.append(f"| {r['cfg']} | {r['protocol']} | {r['universe']} | {r['period']} | {r['oos']} | {_fmt(r['ret'])} | {_fmt(r.get('qqq_eur'))} | {_fmt(r.get('ew_universe_eur'))} | {_fmt(r.get('ew_list07'))} | {_fmt(r['exp_adj_qqq'])} | {_fmt(r.get('random_mean'))} | {_fmt(r['max_dd'])} | {r['time_in_market']*100:.0f}% | {int(r['n_buys'])} | {r['fees']:.0f} |")
        lines.append("")
        lines.append("Retornos de cartera y benchmarks QQQ / EW 87 en EUR (mismo EUR/USD); EW 55 en USD. Universo = Nasdaq-100 de 2026 aplicado al pasado: tanto la estrategia como los EW tienen sesgo de supervivencia; QQQ no.")

    # 6. Universo
    up = os.path.join(R, f"universe_refit{refit_every}.parquet")
    if os.path.exists(up):
        u = pd.read_parquet(up); pers = pd.read_parquet(os.path.join(R, f"universe_persistence_refit{refit_every}.parquet"))
        lines += ["", "## 6. Exclusion de tickers sin mirar el futuro (operaciones C0, semestre siguiente a cada frontera)", "",
                  "| regla | fronteras | excluidos (media) | valor añadido neto/op (media) | positivo en | alpha añadida | B&H lista - B&H todos | percentil vs placebo (R1) |", "|---|---|---|---|---|---|---|---|"]
        for rname, g in u.groupby("rule"):
            lines.append(f"| {rname} | {len(g)} | {g['n_excluded'].mean():.0f} | {_fmt(g['value_added_net'].mean(), d=2)} | {int((g['value_added_net'] > 0).sum())}/{len(g)} | {_fmt(g['value_added_alpha'].mean(), d=2)} | {_fmt((g['bh_inc'] - g['bh_all']).mean())} | {(g['placebo_pct'].mean()*100 if rname=='R1' else np.nan):.0f} |")
        lines += ["", "Persistencia por ticker (alpha 12m previos -> 6m siguientes):", "", "| frontera | n | Spearman | p | tercil inferior sigue inferior |", "|---|---|---|---|---|"]
        for _, r in pers.iterrows():
            lines.append(f"| {r['S']:%Y-%m-%d} | {r['n']} | {r['spearman']:+.2f} | {r['pval']:.2f} | {r['bottom_stays_bottom']*100:.0f}% (base 33%) |")

    # 7. Veredicto mecanico
    lines += ["", "## 7. Veredicto segun reglas pre-registradas", ""]
    c0 = m[(m["key"] == c0k) & (m["period"].isin(FULL))].set_index("period")
    pos_alpha = int((c0["mean_alpha"] > 0).sum())
    beats_qqq = int((c0["mean_qqq_excess"] > 0).sum())
    lines.append(f"- C0 alpha > 0 en {pos_alpha}/5 periodos; neto > QQQ misma ventana en {beats_qqq}/5; SL/TP disparan en {(c0['sh_sl']+c0['sh_tp']).mean()*100:.0f}% de las operaciones.")
    chain_pos = int((chain["oos_alpha"] > 0).sum())
    lines.append(f"- Cadena walk-forward: alpha OOS > 0 en {chain_pos}/{len(chain)} objetivos; la combo elegida supera a C0 en {int((chain['oos_alpha'] > chain['c0_alpha']).sum())}/{len(chain)}.")
    ctrl = summ[(summ["key"] == c0k) & (summ["protocol"] == "post_close") & (summ["screen"] == "False") & (summ["period"].isin(FULL))].pivot_table(index="period", columns="signal", values="mean_alpha")
    if "linreg" in ctrl and "prophet" in ctrl:
        lines.append(f"- Prophet vs regresion lineal (misma regla, sin screening): Prophet mejor en {int((ctrl['prophet'] > ctrl['linreg']).sum())}/5 periodos; vs SMA100 en {int((ctrl['prophet'] > ctrl['sma100']).sum())}/5.")
    if os.path.exists(pp):
        p3 = pf[(pf["cfg"] == "C0") & (pf["protocol"] == "post_close") & (pf["universe"] == "all87") & (pf["slots"] == 3) & (pf["screen"] == "True") & (pf["period"].isin(FULL))]
        lines.append(f"- Cartera C0 (3 slots, 87 tickers): bate a QQQ (EUR) en {int((p3['ret'] > p3['qqq_eur']).sum())}/5, al QQQ ajustado por exposicion en {int((p3['ret'] > p3['exp_adj_qqq']).sum())}/5, al control aleatorio en {int((p3['ret'] > p3['random_mean']).sum())}/5.")
    verdict = []
    wf_beats = None
    if os.path.exists(pp):
        wf = pf3[(pf3["cfg"].str.startswith("WF_")) & (pf3["oos"] == "OOS") & (pf3["universe"] == "all87")]
        if len(wf):
            wf_beats = (int((wf["ret"] > wf["qqq_eur"]).sum()), len(wf))
            lines.append(f"- Cartera de las combos walk-forward en su año OOS (87 tickers, ambos protocolos): bate a QQQ (EUR) en {wf_beats[0]}/{wf_beats[1]} filas.")
    c0_beats_qqq = int((p3["ret"] > p3["exp_adj_qqq"]).sum()) if os.path.exists(pp) else 0
    if os.path.exists(pp) and c0_beats_qqq <= 1 and (wf_beats is None or wf_beats[0] * 2 < wf_beats[1]):
        verdict.append("ABANDONAR la estrategia como esta: ni la config actual ni la mejor combo elegida sin mirar el futuro "
                       "baten a QQQ en cartera; la alpha por operacion es de decimas y los costes a este capital la consumen.")
    elif chain_pos >= 3 and int((chain["oos_alpha"] > chain["c0_alpha"]).sum()) >= 3:
        verdict.append("CAMBIAR a la combo walk-forward (ver seccion 2) - supera a C0 fuera de muestra de forma consistente.")
    elif pos_alpha >= 4:
        verdict.append("MANTENER C0 - alpha positiva en la mayoria de periodos y ninguna alternativa la supera de forma consistente.")
    else:
        verdict.append("SIN EVIDENCIA SUFICIENTE: ni C0 ni las alternativas muestran alpha consistente; no cambiar parametros por ruido.")
    lines += ["", "**" + " ".join(verdict) + "**", ""]
    txt = "\n".join(lines)
    out = os.path.join(R, f"report_refit{refit_every}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"\n[report] -> {out}")
