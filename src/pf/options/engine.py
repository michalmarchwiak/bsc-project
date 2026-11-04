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
    m = ivm.vrp_multiplier(levels.get("VIX", np.nan), sigma_spx)
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


def _positions_for(decisions: pd.DataFrame, policy_name: str, variant: str, lp: bool) -> pd.DataFrame:
    v = variant if lp else "base"
    return decisions[(decisions["policy"] == policy_name) & (decisions["variant"] == v)]


def nav_table(ledger: pd.DataFrame, decisions: pd.DataFrame, ocfg: dict, variants: list[dict] | None = None,
              policies: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stopy okresów i NAV SF dla polityk × wariantów oraz decyzje z wagami i ekspozycją e_i.

    Zwraca (periods: date, end, subfund, policy, variant, r_base, r_ov, nav_base, nav_ov, premium_paid,
    premium_received, kappa_cost, payoff, kind, n_tickers, n_hedged; decisions_out z kolumnami w, e, K, kind)."""
    variants = variants or variant_specs(ocfg)
    sens = set(ocfg.get("sensitivity", {}).get("policies", []))
    pols = policies or list(ocfg["policies"])
    lp_names = set(lp_policies(ocfg))
    icfg = ocfg["implied_vol"]
    rows, dec_out = [], []
    led = {v["variant"]: variant_ledger(ledger, v, icfg) for v in variants}
    for spec in variants:
        Lv = led[spec["variant"]]
        key = ["date", "ticker", "type", "moneyness"]
        prem = Lv.set_index(key)[["premium_rel", "delta", "K"]]
        for d, Ld in Lv.groupby("date"):
            first = Ld.drop_duplicates("ticker").set_index("ticker")
            dd = decisions[decisions["date"] == d]
            for sf, members in first.groupby("subfund"):
                ok = members[np.isfinite(members["S_next"])]
                if ok.empty:
                    continue
                tks = list(ok.index)
                w = pd.Series(1.0 / len(tks), index=tks)
                gg = ok["S_next"] / ok["S_t"]
                ff = ok["fx_next"] / ok["fx_t"]
                GG = gg * ff
                for pname in pols:
                    if spec["variant"] != "base" and pname not in sens:
                        continue
                    pos = _positions_for(dd[dd["subfund"] == sf], pname, spec["variant"], pname in lp_names)
                    pos = pos[pos["ticker"].isin(tks)]
                    if len(pos):
                        j = pos.set_index(key[1:]).index
                        pr = prem.loc[d].reindex(j)
                        pos = pos.assign(premium_rel=pr["premium_rel"].to_numpy(), delta=pr["delta"].to_numpy(),
                                         K=pr["K"].to_numpy())
                    out = nav.period_return(w, GG, gg, ff, pos, spec["kappa"])
                    kind = nav.overlay_kind(pos)
                    rows.append({"date": d, "end": ok["t_next"].iloc[0], "subfund": sf, "policy": pname,
                                 "variant": spec["variant"], **out, "kind": kind, "n_tickers": len(tks),
                                 "n_hedged": int(pos["ticker"].nunique()) if len(pos) else 0})
                    if len(pos) and spec["variant"] == "base":
                        e = nav.delta_exposure(w, pos, spec["kappa"])
                        dec_out.append(pos.assign(w=pos["ticker"].map(w), e=pos["ticker"].map(e), kind=kind))
    periods = pd.DataFrame(rows)
    if len(periods):
        periods = periods.sort_values(["subfund", "policy", "variant", "date"]).reset_index(drop=True)
        grp = periods.groupby(["subfund", "policy", "variant"], sort=False)
        periods["nav_base"] = grp["r_base"].transform(lambda r: np.cumprod(1 + r))
        periods["nav_ov"] = grp["r_ov"].transform(lambda r: np.cumprod(1 + r))
    dec = pd.concat(dec_out, ignore_index=True) if dec_out else pd.DataFrame(columns=DECISION_COLUMNS + ["w", "e"])
    return periods, dec
