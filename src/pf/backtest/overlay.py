"""Backtest nakładki opcyjnej: miary sub-funduszy (P0–P3), okna stresowe, wrażliwość, portfele, raport.

Miary (``pf.backtest.metrics``) na miesięcznych stopach SF z nakładką r_ov i stopie wolnej od ryzyka okresu:
zwrot i zmienność roczna, Sharpe/Sortino, max DD, zrealizowany ES 97,5% stóp miesięcznych, łączny koszt
premii (premie zapłacone − otrzymane, w % NAV na początku okresu, suma), koszt κ, odsetek miesięcy z nakładką
i jej typ. Okna stresowe: skumulowana stopa SF bez nakładki i z nakładką dla okresów z datą t_j w oknie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.backtest import metrics

KINDS = ("put", "covered_call", "collar")


def sample_of(dates: pd.Series, bcfg: dict) -> pd.Series:
    out = pd.Series("other", index=dates.index, dtype=object)
    for name in ("calibration", "holdout"):
        lo, hi = (pd.Timestamp(x) for x in bcfg[name])
        out[(dates >= lo) & (dates <= hi)] = name
    return out


def sf_metrics(periods: pd.DataFrame, alpha: float = 0.975, samples=("all", "calibration", "holdout")) -> pd.DataFrame:
    """Miary per (SF, polityka, wariant, próba); ``periods`` – wynik ``engine.nav_table`` z kolumnami rf, sample."""
    rows = []
    for smp in samples:
        part = periods if smp == "all" else periods[periods["sample"] == smp]
        for (sf, pol, var), g in part.groupby(["subfund", "policy", "variant"]):
            g = g.sort_values("date")
            if len(g) < 3:
                continue
            m = metrics.summary(g["r_ov"], g["rf"], alpha=alpha)
            b = metrics.summary(g["r_base"], g["rf"], alpha=alpha)
            row = {"subfund": sf, "policy": pol, "variant": var, "sample": smp, "T": len(g),
                   "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
                   "sortino": m["sortino"], "max_dd": m["max_dd"], "es_m": m["es_hist"],
                   "ann_return_base": b["ann_return"], "max_dd_base": b["max_dd"], "es_m_base": b["es_hist"],
                   "premium_net": float((g["premium_paid"] - g["premium_received"]).sum()),
                   "kappa_cost": float(g["kappa_cost"].sum()), "share_overlay": float((g["kind"] != "none").mean())}
            row.update({f"share_{k}": float((g["kind"] == k).mean()) for k in KINDS})
            rows.append(row)
    return pd.DataFrame(rows)


def stress_table(periods: pd.DataFrame, windows: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    base = periods[periods["variant"] == "base"]
    for name, (a, b) in windows.items():
        w = base[(base["date"] >= pd.Timestamp(a)) & (base["date"] <= pd.Timestamp(b))]
        for (sf, pol), g in w.groupby(["subfund", "policy"]):
            rb, ro = float(np.prod(1 + g["r_base"]) - 1), float(np.prod(1 + g["r_ov"]) - 1)
            rows.append({"window": name, "subfund": sf, "policy": pol, "n_months": len(g), "ret_base": rb,
                         "ret_ov": ro, "diff": ro - rb, "worst_month_base": float(g["r_base"].min()),
                         "worst_month_ov": float(g["r_ov"].min())})
    return pd.DataFrame(rows)


def portfolio_metrics(pp: pd.DataFrame, gammas: dict[str, float], alpha: float = 0.975,
                      samples=("all", "calibration", "holdout")) -> pd.DataFrame:
    rows = []
    for smp in samples:
        part = pp if smp == "all" else pp[pp["sample"] == smp]
        for (pid, strat, var), g in part.groupby(["profile", "strategy", "variant"]):
            g = g.sort_values("date")
            if len(g) < 3:
                continue
            for col, label in (("r_net", "raw"), ("r_es_matched", "es_matched")):
                m = metrics.summary(g[col], g["rf"], gamma=gammas.get(pid), alpha=alpha)
                rows.append({"profile": pid, "strategy": strat, "variant": var, "series": label, "sample": smp,
                             "T": len(g), **{k: m[k] for k in ("ann_return", "ann_vol", "sharpe", "sortino", "max_dd",
                                                                 "es_hist", "ce_ann")},
                             "es_exante_mean": float(g["es_exante"].mean()), "turnover": float(g["turnover"].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- raport

def _pct(x, d=2) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}".replace(".", ",")


def _num(x, d=2) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace(".", ",")


def report(sfm: pd.DataFrame, stress: pd.DataFrame, pfm: pd.DataFrame | None, validation: pd.DataFrame | None,
           annual: pd.DataFrame | None, meta: dict) -> str:
    L = ["# Miesięczna nakładka opcyjna – backtest i walidacja", "",
         "Opcje są **hipotetyczne**: dla większości tickerów (spółki GPW, ETF-y, złoto) nie istnieją notowane opcje "
         "miesięczne, a σ_imp jest odtwarzana modelowo (GVZ dla ETF-ów na złoto, w pozostałych przypadkach "
         "σ̂_GARCH × m_t, m_t = VIX/σ̂_SPX). Opcje europejskie bliskie ATM trzymane do wygaśnięcia, bez "
         "przeszacowania w trakcie miesiąca. Szczegóły: `docs/options_monthly.md`.", "",
         f"Okres: {meta.get('first_date')} – {meta.get('last_date')} ({meta.get('n_dates')} dat). "
         f"κ = {_pct(meta.get('kappa'), 1)}% premii, pasmo |ln(K/F)| ≤ {_num(meta.get('band'), 2)}, "
         f"m̄ = {_num(meta.get('m_bar'), 3)}, atm_adj = {_num(meta.get('atm_adj'), 2)}, "
         f"polityka SFk_OV: {meta.get('selected')}.", ""]
    base = sfm[(sfm["variant"] == "base") & (sfm["sample"] == "all")]
    L += ["## Sub-fundusze: polityki P0–P3 (cała próba, wariant bazowy)", "",
          "Zwrot i zmienność roczne [%], ES – zrealizowany ES 97,5% stóp miesięcznych [%], premia netto i κ – suma "
          "w % NAV, udział – odsetek miesięcy z nakładką (put / covered call / collar).", "",
          "| SF | polityka | zwrot | zmienn. | Sharpe | Sortino | max DD | ES mies. | premia netto | κ | nakładka | put/cc/collar |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in base.sort_values(["subfund", "policy"]).itertuples():
        L.append(f"| {r.subfund} | {r.policy} | {_pct(r.ann_return)} | {_pct(r.ann_vol)} | {_num(r.sharpe)} | "
                 f"{_num(r.sortino)} | {_pct(r.max_dd)} | {_pct(r.es_m)} | {_pct(r.premium_net, 1)} | "
                 f"{_pct(r.kappa_cost, 1)} | {_pct(r.share_overlay, 0)} | {_pct(r.share_put, 0)}/"
                 f"{_pct(r.share_covered_call, 0)}/{_pct(r.share_collar, 0)} |")
    if len(stress):
        L += ["", "## Okna stresowe (skumulowana stopa SF, %)", "",
              "| okno | SF | polityka | mies. | bez nakładki | z nakładką | różnica |", "|---|---|---|---:|---:|---:|---:|"]
        for r in stress[stress["policy"] != "P0"].sort_values(["window", "subfund", "policy"]).itertuples():
            L.append(f"| {r.window} | {r.subfund} | {r.policy} | {r.n_months} | {_pct(r.ret_base)} | {_pct(r.ret_ov)} | "
                     f"{_pct(r.diff)} |")
    sens = sfm[(sfm["sample"] == "all") & (sfm["variant"] != "base")]
    if len(sens):
        ref = base.set_index(["subfund", "policy"])["ann_return"]
        avg = sens.assign(d=sens["ann_return"] - sens.set_index(["subfund", "policy"]).index.map(ref)) \
            .groupby(["policy", "variant"])["d"].mean().reset_index()
        L += ["", "## Wrażliwość (średnia po SF zmiany zwrotu rocznego względem wariantu bazowego, p.p.)", "",
              "| polityka | wariant | Δ zwrot |", "|---|---|---:|"]
        L += [f"| {r.policy} | {r.variant} | {_pct(r.d)} |" for r in avg.itertuples()]
    if pfm is not None and len(pfm):
        L += ["", "## Portfele P3–P5: bez nakładki (base) i z wariantami SFk_OV (ov)", "",
              "„es_matched” – oba warianty sprowadzone do tego samego ES ex ante (domieszka lokaty po r_f).", "",
              "| próba | profil | strategia | wariant | seria | zwrot | zmienn. | Sharpe | max DD | ES mies. | CE |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in pfm.sort_values(["sample", "profile", "strategy", "series", "variant"]).itertuples():
            L.append(f"| {r.sample} | {r.profile} | {r.strategy} | {r.variant} | {r.series} | {_pct(r.ann_return)} | "
                     f"{_pct(r.ann_vol)} | {_num(r.sharpe)} | {_pct(r.max_dd)} | {_pct(r.es_hist)} | {_pct(r.ce_ann)} |")
    if validation is not None and len(validation):
        v = validation[validation["period"].isin(["all", "stress_2008", "stress_2020", "stress_2022"])]
        L += ["", "## Walidacja wyceny: replikacja Cboe BXM i PPUT", "",
              "TE = σ(r_rep − r_off)·√12 na stopach między rolowaniami (3. piątek).", "",
              "| indeks | wariant | okres | mies. | TE [% p.a.] | śr. różnica [% p.a.] | korelacja |",
              "|---|---|---|---:|---:|---:|---:|"]
        L += [f"| {r.index} | {r.variant} | {r.period} | {r.n_months} | {_num(r.te_ann_pct)} | "
              f"{_num(r.mean_diff_ann_pct)} | {_num(r.corr, 3)} |" for r in v.itertuples()]
    if annual is not None and len(annual):
        L += ["", f"## Scenariusze roczne z nakładką miesiąc po miesiącu (t = {meta.get('annual_date')})", "",
              "| SF | polityka | średnia [%] | VaR 97,5% [%] | ES 97,5% [%] | P(R < 0) |", "|---|---|---:|---:|---:|---:|"]
        L += [f"| {r.subfund} | {r.policy} | {_pct(r.mean)} | {_pct(r.var)} | {_pct(r.es)} | {_pct(r.p_loss, 1)} |"
              for r in annual.sort_values(["subfund", "policy"]).itertuples()]
    return "\n".join(L) + "\n"
