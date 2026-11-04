"""Silnik nakładki opcyjnej: krok daty t_j (rejestr, σ_imp, decyzje P3) i konsolidacja (P0–P2, NAV, ekspozycje).

Krok daty (``date_step``), dane ≤ t_j:
    1. model GARCH-kopuła z okna ≤ t_j − 1 (``pf.options.scenarios``), margines S&P 500 → m_t;
    2. σ̂_i(τ), σ_imp (``pf.options.implied_vol``), stopy, ceny i kursy z zamknięcia t_j;
    3. rejestr opcji całej siatki (``pf.options.ledger``) z rozliczeniem w t_{j+1};
    4. scenariusze S z dryfem konserwatywnym i decyzje P3 (LP) dla każdego SF, wariantu λ i wariantu
       wrażliwości (σ_imp ± p.p. dla tickerów proxy, β, κ).

Konsolidacja (``consolidate``): sygnały P2 z dat ≤ t_j (percentyle w oknie rozszerzającym się), pozycje P0–P2,
tabela decyzji z ekspozycją skorygowaną o deltę oraz NAV każdego SF dla każdej polityki i wariantu.
Wagi wewnętrzne SF: równe między tickerami SF dostępnymi w t_j (reguła macierzy A, ``pf.portfolio``).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pf.data.usd_rate import continuous_rate
from pf.options import implied_vol as ivm
from pf.options import ledger as ledger_mod
from pf.options import nav, policy, scenarios
from pf.options.market import MarketData, currency
from pf.portfolio import deterministic_tickers

DECISION_COLUMNS = ["date", "subfund", "policy", "variant", "ticker", "type", "moneyness", "h", "premium_rel",
                    "delta"]
VOL_NAMES = ("VIX", "VIX9D", "VIX3M", "GVZ", "OVX")


# --------------------------------------------------------------------------- konfiguracja

def overlay_subfunds(subfunds: dict, ocfg: dict) -> dict[str, list[str]]:
    """SF objęte nakładką (bez ``exclude_subfunds`` i SF deterministycznych) → tickery."""
    excl = set(ocfg.get("exclude_subfunds", []))
    return {sf: list(spec["tickers"]) for sf, spec in subfunds.items()
            if sf not in excl and not spec.get("deterministic", False)}


def lp_policies(ocfg: dict) -> dict[str, dict]:
    return {k: v for k, v in ocfg["policies"].items() if v["kind"] == "lp"}


def variant_specs(ocfg: dict) -> list[dict]:
    """Wariant bazowy i warianty wrażliwości: σ_imp ± p.p. (proxy), β, κ."""
    base = {"variant": "base", "shift_pp": 0.0, "beta": float(ocfg["implied_vol"].get("beta", 0.0)),
            "kappa": float(ocfg["kappa"])}
    out = [base]
    s = ocfg.get("sensitivity", {})
    if s.get("enabled", False):
        pp = float(s.get("sigma_shift_pp", 0.0))
        if pp:
            out += [{**base, "variant": f"sigma+{pp:g}pp", "shift_pp": pp},
                    {**base, "variant": f"sigma-{pp:g}pp", "shift_pp": -pp}]
        if s.get("beta") is not None:
            out.append({**base, "variant": f"beta={float(s['beta']):g}", "beta": float(s["beta"])})
        for k in s.get("kappa", []):
            out.append({**base, "variant": f"kappa={float(k):g}", "kappa": float(k)})
    return out


def variant_ledger(L: pd.DataFrame, spec: dict, icfg: dict) -> pd.DataFrame:
    """Rejestr w wariancie wrażliwości (ponowna wycena przy zmienionym σ_ATM tickerów proxy lub β)."""
    base_beta = float(icfg.get("beta", 0.0))
    if spec["shift_pp"] == 0.0 and spec["beta"] == base_beta:
        return L
    sig = L["sigma_atm"].to_numpy(float)
    if spec["shift_pp"]:
        proxy = (L["iv_source"] == "proxy").to_numpy()
        sig = np.where(proxy, np.clip(sig + spec["shift_pp"] / 100.0, icfg["sigma_min"], icfg["sigma_max"]), sig)
    return ledger_mod.reprice(L, sig, spec["beta"], icfg["sigma_min"])


# --------------------------------------------------------------------------- krok daty

def date_step(period: dict, market: MarketData, cfg: dict, ocfg: dict, subfunds: dict, seed: int,
              cache_dir=None, jobs: int | None = 1, model=None) -> dict:
    """Rejestr, stan rynku, σ_imp tickerów i decyzje P3 w dacie t_j (``period`` – wiersz ``overlay_periods``)."""
    t0 = time.time()
    t = pd.Timestamp(period["date"])
    skip, n, tau, days = int(period["skip"]), int(period["n_sessions"]), float(period["tau"]), float(period["days"])
    icfg, lcfg = ocfg["implied_vol"], ocfg["lp"]
    td = int(ocfg.get("trading_days", 252))
    sf_tickers = overlay_subfunds(subfunds, ocfg)
    subfund_of = {tk: sf for sf, tks in sf_tickers.items() for tk in tks}
    all_tickers = [tk for spec in subfunds.values() for tk in spec["tickers"]]
    if model is None:
        model = scenarios.fit_overlay_model(market, t, cfg, ocfg, all_tickers, deterministic_tickers(subfunds),
                                            cache_dir, jobs)
    t_open, t_close = pd.Timestamp(period["open"]), period["close"]
    tickers = [a for a in model.assets if a in subfund_of]
    S_t = market.spot(t_open, tickers)
    tickers = [tk for tk in tickers if np.isfinite(S_t.get(tk, np.nan))]
    S_t = S_t.reindex(tickers)
    levels = {v: market.vol_level(v, t_open) for v in VOL_NAMES}
    sigma_spx = ivm.garch_vol(model.spx, int(icfg.get("m_horizon", 21)), skip, td) if model.spx else float("nan")
    m = ivm.market_multiplier(levels.get("VIX", np.nan), sigma_spx, icfg)
    sig = ivm.sigma_table(tickers, model.margins, n, skip, m, levels, days, icfg, td)
    fx_t = market.fx_of(t_open, tickers)
    if pd.notna(t_close):
        S_next, fx_next = market.spot(t_close, tickers), market.fx_of(t_close, tickers)
    else:
        S_next = fx_next = pd.Series(np.nan, index=tickers)
    y_pln, y_usd = market.rate("PLN", t_open), market.rate("USD", t_open)
    r = ledger_mod.period_rates(tickers, y_pln, y_usd, days, tau, ocfg["rates"])
    L = ledger_mod.build(period, tickers, subfund_of, S_t, S_next, fx_t, fx_next, r, sig, ocfg)
    t_led = time.time()

    # scenariusze (miara P, dryf konserwatywny) i decyzje P3
    scen = scenarios.simulate(model, int(lcfg["n_scenarios"]), n, skip, seed, lcfg.get("seed_tag", "options_overlay"),
                              cfg["margins"].get("sim_eps_cap"))
    rc_pln = float(continuous_rate(y_pln, days, tau, ocfg["rates"].get("pln_basis", 365)))
    rc_usd = float(continuous_rate(y_usd, days, tau, ocfg["rates"].get("usd_basis", 360)))
    g, f, _ = scen.panels(tickers)
    g, f = scenarios.drift_adjust(g, f, tickers, r, sig["sigma_hat"], tau, float(lcfg["drift_sharpe"]), rc_pln, rc_usd)
    G = g * f
    col = {tk: i for i, tk in enumerate(tickers)}
    dec_rows, info_rows = [], []
    lp_pol = lp_policies(ocfg)
    sens_pol = set(ocfg.get("sensitivity", {}).get("policies", []))
    for spec in variant_specs(ocfg):
        Lv = variant_ledger(L, spec, icfg)
        legs = policy.candidate_legs(Lv, ocfg)
        for pname, pspec in lp_pol.items():
            if spec["variant"] != "base" and pname not in sens_pol:
                continue
            for sf, members in sf_tickers.items():
                tks = [tk for tk in members if tk in col]
                if not tks:
                    continue
                idx = [col[tk] for tk in tks]
                w = np.full(len(tks), 1.0 / len(tks))
                pos, info = policy.lp_positions(w, G[:, idx], g[:, idx], f[:, idx], tks, legs, spec["kappa"],
                                                float(pspec["lambda"]), float(lcfg["alpha"]), float(lcfg["epsilon"]))
                for p in pos.itertuples(index=False):
                    dec_rows.append({"date": t, "subfund": sf, "policy": pname, "variant": spec["variant"],
                                     **p._asdict()})
                info_rows.append({"date": t, "subfund": sf, "policy": pname, "variant": spec["variant"],
                                  "kind": nav.overlay_kind(pos), **{k: v for k, v in info.items() if k != "zeta"}})
    state = {"date": t, "open": t_open, "close": t_close, "n_sessions": n, "tau": tau, "days": days, "m": m,
             "sigma_spx": sigma_spx, **{f"lvl_{k}": v for k, v in levels.items()}, "rate_pln": y_pln,
             "rate_usd": y_usd, "n_tickers": len(tickers), "dep_model": model.dep.name,
             "nu": float(getattr(model.dep, "nu", np.nan)), "model_cached": model.cached,
             "n_refit": len(getattr(model, "refit", [])), "refit": ",".join(getattr(model, "refit", [])),
             "n_scen_dropped": scen.n_dropped,
             "sec_fit": model.timings.get("fit", 0.0), "sec_ledger": t_led - t0, "sec_sim": scen.seconds,
             "sec_total": time.time() - t0}
    tick = sig.reset_index().assign(date=t, subfund=lambda d: d["ticker"].map(subfund_of),
                                    S_t=S_t.reindex(sig.index).to_numpy(),
                                    currency=lambda d: d["ticker"].map(currency))
    return {"ledger": L, "state": pd.DataFrame([state]), "tickers": tick,
            "decisions": pd.DataFrame(dec_rows, columns=DECISION_COLUMNS),
            "lp_info": pd.DataFrame(info_rows), "model": model}


# --------------------------------------------------------------------------- konsolidacja

def p012_decisions(ledger: pd.DataFrame, signals: pd.DataFrame, ocfg: dict) -> pd.DataFrame:
    """Pozycje P0–P2 (wariant bazowy; w wariantach wrażliwości te same h, inne premie)."""
    rows = []
    for d, Ld in ledger.groupby("date"):
        legs = policy.candidate_legs(Ld, ocfg)
        sig_d = signals[signals["date"] == d]
        for pname, spec in ocfg["policies"].items():
            if spec["kind"] == "static":
                pos = policy.static_positions(legs, list(Ld["ticker"].unique()), spec)
            elif spec["kind"] == "rules":
                pos = policy.rule_positions(legs, sig_d, ocfg["rules"])
            else:
                continue
            if pos.empty:
                continue
            pos = pos.assign(date=d, policy=pname, variant="base",
                             subfund=pos["ticker"].map(Ld.drop_duplicates("ticker").set_index("ticker")["subfund"]))
            rows.append(pos)
    if not rows:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    return pd.concat(rows, ignore_index=True)[DECISION_COLUMNS]


def signals_table(state: pd.DataFrame, tick: pd.DataFrame, market: MarketData, ocfg: dict) -> pd.DataFrame:
    """Sygnały P2 w datach t_j (wyłącznie z dat ≤ t_j)."""
    sc = ocfg["signals"]
    opens = state.set_index("date")["open"].sort_index()
    tickers = sorted(tick["ticker"].unique())
    prices = pd.DataFrame({d: market.spot(o, tickers) for d, o in opens.items()}).T
    return policy.rule_signals(state[["date", "m"]], tick[["date", "ticker", "sigma_hat"]], prices,
                               int(sc["min_history_months"]), int(sc["sma_months"]))


def _variant_positions(decisions: pd.DataFrame, Lv: pd.DataFrame, variant: str, pols: list[str],
                       lp_names: set[str]) -> dict[tuple, pd.DataFrame]:
    """Pozycje polityk w wariancie z premiami z rejestru wariantu: (date, SF, polityka) → nogi.

    Polityki LP mają decyzje własne dla wariantu, pozostałe – decyzje bazowe (te same h, inne premie)."""
    lp = decisions["policy"].isin(lp_names)
    sel = decisions["policy"].isin(pols) & ((lp & (decisions["variant"] == variant)) |
                                            (~lp & (decisions["variant"] == "base")))
    d = decisions[sel].drop(columns=["premium_rel", "delta", "K"], errors="ignore")
    if d.empty:
        return {}
    key = ["date", "ticker", "type", "moneyness"]
    d = d.merge(Lv[key + ["premium_rel", "delta", "K"]], on=key, how="left")
    out = {}
    for k, g in d.groupby(["date", "subfund", "policy"], sort=False):
        out[k] = (g.reset_index(drop=True), g["ticker"].to_numpy(object), (g["type"] == "call").to_numpy(bool),
                  *(g[c].to_numpy(float) for c in ("moneyness", "h", "premium_rel", "delta")))
    return out


def nav_table(ledger: pd.DataFrame, decisions: pd.DataFrame, ocfg: dict, variants: list[dict] | None = None,
              policies: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stopy okresów i NAV SF dla polityk × wariantów oraz decyzje z wagami i ekspozycją e_i.

    Zwraca (periods: date, end, subfund, policy, variant, r_base, r_ov, nav_base, nav_ov, premium_paid,
    premium_received, kappa_cost, payoff, kind, n_tickers, n_hedged; decisions_out z kolumnami w, e, K, kind)."""
    variants = variants or variant_specs(ocfg)
    sens = set(ocfg.get("sensitivity", {}).get("policies", []))
    pols_all = policies or list(ocfg["policies"])
    lp_names = set(lp_policies(ocfg))
    icfg = ocfg["implied_vol"]
    rows, dec_out = [], []
    firsts = {d: g.drop_duplicates("ticker").set_index("ticker") for d, g in ledger.groupby("date")}
    for spec in variants:
        base_variant = spec["variant"] == "base"
        pols = pols_all if base_variant else [p for p in pols_all if p in sens]
        Lv = variant_ledger(ledger, spec, icfg)
        posd = _variant_positions(decisions, Lv, spec["variant"], pols, lp_names)
        for d, first in firsts.items():
            for sf, members in first.groupby("subfund", observed=True):
                ok = members[np.isfinite(members["S_next"])]
                if ok.empty:
                    continue
                tks = list(ok.index)
                tpos = {t: i for i, t in enumerate(tks)}
                wv = np.full(len(tks), 1.0 / len(tks))
                gg = (ok["S_next"] / ok["S_t"]).to_numpy(float)
                ff = (ok["fx_next"] / ok["fx_t"]).to_numpy(float)
                GG = gg * ff
                end = ok["t_next"].iloc[0]
                for pname in pols:
                    grp = posd.get((d, sf, pname))
                    if grp is None:
                        ix, keep, arr = np.zeros(0, int), np.zeros(0, bool), [np.zeros(0)] * 4
                        is_call = np.zeros(0, bool)
                    else:
                        frame, tk, is_call, *arr = grp
                        ix = np.array([tpos.get(t, -1) for t in tk], dtype=int)
                        keep = (ix >= 0) & (arr[1] > 0)
                        ix, is_call, arr = ix[keep], is_call[keep], [a[keep] for a in arr]
                    out = nav.period_arrays(wv, GG, gg, ff, ix, is_call, *arr, spec["kappa"])
                    e, kind = out.pop("e"), out.pop("kind")
                    rows.append({"date": d, "end": end, "subfund": sf, "policy": pname, "variant": spec["variant"],
                                 **out, "kind": kind, "n_tickers": len(tks), "n_hedged": int(len(np.unique(ix)))})
                    if len(ix) and base_variant:
                        dec_out.append(frame[keep].assign(w=1.0 / len(tks), e=e[ix], kind=kind))
    periods = pd.DataFrame(rows)
    if len(periods):
        periods = periods.sort_values(["subfund", "policy", "variant", "date"]).reset_index(drop=True)
        grp = periods.groupby(["subfund", "policy", "variant"], sort=False)
        periods["nav_base"] = grp["r_base"].transform(lambda r: np.cumprod(1 + r))
        periods["nav_ov"] = grp["r_ov"].transform(lambda r: np.cumprod(1 + r))
    dec = pd.concat(dec_out, ignore_index=True) if dec_out else pd.DataFrame(columns=DECISION_COLUMNS + ["w", "e"])
    return periods, dec


# --------------------------------------------------------------------------- integracja z portfelem (Etap 5)

def positions_with_premia(decisions: pd.DataFrame, ledger_t: pd.DataFrame, policy_name: str) -> dict[str, pd.DataFrame]:
    """Pozycje polityki w dacie (wariant bazowy) z premiami z rejestru: SF → DataFrame nóg."""
    d = decisions[(decisions["policy"] == policy_name) & (decisions["variant"] == "base")]
    if d.empty:
        return {}
    key = ["ticker", "type", "moneyness"]
    prem = ledger_t.set_index(key)[["premium_rel", "delta"]]
    d = d.drop(columns=["premium_rel", "delta"], errors="ignore").join(prem, on=key)
    return {sf: g[policy.LEG_COLUMNS].reset_index(drop=True) for sf, g in d.groupby("subfund") if len(g)}


def portfolio_step(period: dict, market: MarketData, model, cfg: dict, ocfg: dict, pcfg: dict, subfunds: dict,
                   positions: dict[str, pd.DataFrame], seed: int) -> dict:
    """Wagi poziomów 1–2 bez i z kolumnami SFk_OV dla profili i strategii z ``monthly_overlay.portfolio``."""
    from pf.opt.profiles import load_profiles, subfund_data
    from pf.options import portfolio as port

    t0 = time.time()
    t = pd.Timestamp(period["date"])
    pc = ocfg["portfolio"]
    alpha = float(cfg["rebalance"]["es_alpha"])
    scen = scenarios.simulate(model, int(pc["n_scenarios"]), int(period["n_sessions"]), int(period["skip"]), seed,
                              ocfg["lp"].get("seed_tag", "options_overlay"), cfg["margins"].get("sim_eps_cap"))
    assets = [a for a in model.assets if a != scenarios.FX]
    g, f, G = scen.panels(assets)
    R = G - 1.0
    sfs_all = {sf: spec for sf, spec in subfunds.items() if set(spec["tickers"]) & set(assets)}
    sfs = subfund_data(R, assets, sfs_all)
    gross = {a: (g[:, k], f[:, k]) for k, a in enumerate(assets)}
    profiles = load_profiles(pcfg)
    opt = pcfg.get("optimizer", {})
    measure, q = pcfg["dependence_measure"], float(pcfg["tail_q"])
    ov_sfs = set(overlay_subfunds(subfunds, ocfg))
    wrows, lrows, xrows = [], [], []
    cache: dict = {}
    for pid in pc["profiles"]:
        prof = profiles[pid]
        for strat in pc["strategies"]:
            pos = positions if port.ov_allowed(pid, ocfg) else {}
            res = port.optimize_with_overlay(sfs, gross, pos, prof, strat, alpha, opt, float(ocfg["kappa"]), measure,
                                             q, cache, ov_sfs)
            if port.ov_allowed(pid, ocfg) and "ov" not in res:
                res["ov"] = res["base"]          # brak aktywnej nakładki w t_j: SFk_OV ≡ SFk
            for variant, (r2, Rc, names) in res.items():
                W = r2.W.reindex(names).to_numpy(float)
                ea = port.ex_ante(Rc, W, alpha, prof.gamma)
                xrows.append({"date": t, "profile": pid, "strategy": strat, "variant": variant, **ea,
                              "n_ov_columns": int(sum(nm.endswith(port.OV_SUFFIX) for nm in names)),
                              "w_ov": float(sum(w for nm, w in zip(names, W) if nm.endswith(port.OV_SUFFIX))),
                              "lt_relaxed": bool(r2.flags.get("lt_relaxed", False))})
                wrows += [{"date": t, "profile": pid, "strategy": strat, "variant": variant, "column": nm, "W": w}
                          for nm, w in zip(names, W)]
            base = res["base"][0]
            lrows += [{"date": t, "profile": pid, "strategy": strat, "subfund": sf, "ticker": tk, "w": wv}
                      for sf, ws in base.w.items() for tk, wv in ws.items() if wv > 1e-12]
    return {"weights": pd.DataFrame(wrows), "level1": pd.DataFrame(lrows), "exante": pd.DataFrame(xrows),
            "seconds": time.time() - t0}


def portfolio_realize(periods: pd.DataFrame, market: MarketData, weights: pd.DataFrame, level1: pd.DataFrame,
                      exante: pd.DataFrame, positions_by_date: dict, ocfg: dict, cost: float, lag: int,
                      day_count: int = 365) -> pd.DataFrame:
    """Zrealizowane stopy portfeli (brutto, netto po kosztach obrotu akcji, wariant o tym samym ÊS).

    Koszt obrotu: c·Σ_i |x_{t,i} − x̃_{t,i}| na ekspozycji akcyjnej look-through (SFk: W_k w_ki,
    SFk_OV: W_k^OV w_ki / Σ w D), x̃ – ekspozycja po dryfie poprzedniego okresu; koszt κ premii jest w stopie SFk_OV."""
    from pf.data.riskfree import period_rate
    from pf.options import calendar as cal
    from pf.options import portfolio as port

    kappa = float(ocfg["kappa"])
    rows = []
    prev_drift: dict[tuple, pd.Series] = {}
    w1 = {k: g for k, g in level1.groupby(["date", "profile", "strategy"])}
    xa = exante.set_index(["date", "profile", "strategy", "variant"])["es"]
    for per in periods.sort_values("date").to_dict("records"):
        if pd.isna(per["t_next"]):
            continue
        d = per["date"]
        Wd = weights[weights["date"] == d]
        if Wd.empty:
            continue
        sessions = cal.sessions_of(market.index, per, lag)
        tick_all = sorted(level1.loc[level1["date"] == d, "ticker"].unique())
        Gr = market.gross(sessions, tick_all, pln=True)
        gr = market.gross(sessions, tick_all, pln=False)
        fr = Gr / gr
        rf = period_rate(market.rate_pln, per["open"], per["close"], day_count)
        pos_d = positions_by_date.get(d, {})
        for (pid, strat, variant), Wg in Wd.groupby(["profile", "strategy", "variant"]):
            l1 = w1.get((d, pid, strat))
            if l1 is None:
                continue
            wsf = {sf: g.set_index("ticker")["w"] for sf, g in l1.groupby("subfund")}
            r_p, x, end_val = 0.0, pd.Series(0.0, index=tick_all), pd.Series(0.0, index=tick_all)
            for col, W in zip(Wg["column"], Wg["W"]):
                if W <= 0:
                    continue
                sf = col[:-len(port.OV_SUFFIX)] if col.endswith(port.OV_SUFFIX) else col
                w = wsf[sf]
                if col.endswith(port.OV_SUFFIX):
                    out = nav.period_return(w, Gr, gr, fr, pos_d.get(sf), kappa)
                    r_col, stock = out["r_ov"], w / out["denominator"]
                else:
                    r_col, stock = float(w @ Gr.reindex(w.index)) - 1.0, w
                r_p += W * r_col
                x = x.add(W * stock, fill_value=0.0)
                end_val = end_val.add(W * stock * Gr.reindex(stock.index), fill_value=0.0)
            key = (pid, strat, variant)
            prev = prev_drift.get(key)
            traded = float(x.sum()) if prev is None else float(x.sub(prev, fill_value=0.0).abs().sum())
            k = cost * traded
            prev_drift[key] = end_val / (1.0 + r_p)
            r_net = (1.0 - k) * (1.0 + r_p) - 1.0
            es_v = xa.get((d, pid, strat, variant), np.nan)
            es_other = [xa.get((d, pid, strat, v), np.nan) for v in ("base", "ov")]
            a = port.es_scale(es_v, np.nanmin(es_other), rf) if np.isfinite(es_other).all() else 1.0
            rows.append({"date": d, "end": per["t_next"], "profile": pid, "strategy": strat, "variant": variant,
                         "r_gross": r_p, "r_net": r_net, "cost": k, "turnover": 0.5 * traded, "rf": rf,
                         "es_exante": es_v, "es_scale": a, "r_es_matched": a * r_net + (1 - a) * rf})
    return pd.DataFrame(rows)
