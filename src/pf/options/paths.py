"""Scenariusze wielomiesięczne (np. roczne) sub-funduszy z nakładką stosowaną miesiąc po miesiącu.

Ścieżka s startuje ze stanu modelu w t_j (``pf.options.scenarios.OverlayModel``): dla każdego miesiąca
m = 1, …, M (n sesji):
    1. σ̂⁽ˢ⁾_{i,m} = √[(252/n) Σ_{h=1}^{n} E σ²_{i,h}] z rekursji GARCH w ścieżce (stan σ² na początku miesiąca);
    2. σ_imp⁽ˢ⁾ = clip(m̄ · σ̂⁽ˢ⁾, σ_min, σ_max), m̄ – mediana historyczna m_t (dane ≤ t_j);
    3. premie BSM nóg polityki (q = 0, stopy z t_j stałe w horyzoncie);
    4. n dni ścieżki z kopuły i GJR-GARCH, brutto-stopy g (waluta notowania) i f (USD/PLN);
    5. stopa SF z nakładką (``pf.options.nav``, wagi równe, finansowanie premii z kapitału SF).
Stopa w horyzoncie: Π_m (1 + R_m) − 1.

Polityki: P0, P1 / P1_cc (statyczne), P2 (reguły: Q – percentyl m̄ w historii m_t, T – trend ze ścieżki cen
i historii, V – percentyl σ̂⁽ˢ⁾ w historii σ̂_i) oraz ``P3_frozen`` – średnie hedge ratio z backtestu P3
(dane ≤ t_j) stosowane bez rozwiązywania LP w ścieżkach (koszt symulacji zagnieżdżonej).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pf.models.margins import MarginFit, skewt_ppf
from pf.options import bs
from pf.options.implied_vol import persistence, variance_path
from pf.options.scenarios import FX, OverlayModel
from pf.options.market import currency
from pf.sim.scenarios import full_uniforms, model_seed


class PathState:
    """Stan rekursji AR(1)–GJR-GARCH(1,1) w ścieżkach: σ² następnej sesji i ostatnia stopa (×scale)."""

    def __init__(self, fits: list[MarginFit], S: int):
        self.fits = fits
        self.par = {k: np.array([getattr(f, k) for f in fits]) for k in
                    ("mu", "phi", "omega", "alpha", "gamma", "beta", "scale")}
        self.sigma2 = np.tile(np.array([f.next_sigma2() for f in fits]), (S, 1))
        self.r_prev = np.tile(np.array([f.last_r for f in fits]), (S, 1))
        self.rho = np.array([persistence(f) if not f.deterministic else 0.0 for f in fits])
        self.det = np.array([f.deterministic for f in fits])

    def step_block(self, U: np.ndarray, eps_cap: float | None) -> np.ndarray:
        """Suma log-stóp (jednostki pierwotne) z bloku dni U (S, n, d); aktualizuje stan."""
        S, n, d = U.shape
        Z = np.zeros_like(U)
        for j, f in enumerate(self.fits):
            if not f.deterministic:
                Z[:, :, j] = skewt_ppf(U[:, :, j].ravel(), f.eta, f.lam).reshape(S, n)
        cap = None
        if eps_cap is not None:
            cap = np.array([eps_cap * f.eps_absmax if np.isfinite(f.eps_absmax) else np.inf for f in self.fits])
        p = self.par
        total = np.zeros((S, d))
        for h in range(n):
            eps = np.sqrt(self.sigma2) * Z[:, h, :]
            if cap is not None:
                eps = np.clip(eps, -cap, cap)
            r = p["mu"] + p["phi"] * self.r_prev + eps
            total += r
            self.sigma2 = p["omega"] + (p["alpha"] + p["gamma"] * (eps < 0)) * eps ** 2 + p["beta"] * self.sigma2
            self.r_prev = r
        return total / p["scale"]

    def avg_vol(self, n: int, trading_days: int = 252) -> np.ndarray:
        """σ̂⁽ˢ⁾ (S, d) – prognoza średniej zmienności na kolejne n sesji ze stanu ścieżki."""
        v = variance_path(self.sigma2, self.par["omega"], self.rho, n)
        return np.sqrt(trading_days / n * v.sum(axis=-1)) / self.par["scale"]


def _percentile(hist: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentyl x względem historii (empiryczna dystrybuanta z x dołączonym do próby)."""
    hist = np.sort(np.asarray(hist, dtype=float)[np.isfinite(hist)])
    if len(hist) == 0:
        return np.full(np.shape(x), np.nan)
    return (np.searchsorted(hist, x, side="right") + 1) / (len(hist) + 1)


def simulate_overlay_paths(model: OverlayModel, sf_tickers: dict[str, list[str]], policies: dict[str, dict],
                           rates: pd.Series, m_bar: float, months: int, n: int, n_scenarios: int, seed: int,
                           ocfg: dict, history: dict | None = None, frozen: pd.DataFrame | None = None,
                           eps_cap: float | None = 2.0) -> pd.DataFrame:
    """Stopy SF w horyzoncie ``months`` miesięcy dla polityk (S wierszy × kolumny SF|polityka).

    ``rates`` – stopa ciągła r_i tickerów (na rok); ``history`` – {'m': szereg m_t, 'sigma_hat': DataFrame σ̂
    (daty × tickery), 'prices': DataFrame cen w datach t_j} do reguł P2; ``frozen`` – średnie h P3
    (ticker, type, moneyness, h)."""
    icfg, td, kappa = ocfg["implied_vol"], int(ocfg.get("trading_days", 252)), float(ocfg["kappa"])
    margins = [model.margins[a] for a in model.assets]
    state = PathState(margins, n_scenarios)
    pos = {a: i for i, a in enumerate(model.assets)}
    tickers = [t for tks in sf_tickers.values() for t in tks if t in pos]
    ti = np.array([pos[t] for t in tickers])
    usd = np.array([currency(t) == "USD" for t in tickers])
    r_i = rates.reindex(tickers).to_numpy(float)
    tau = n / td
    w_rng = np.random.default_rng(model_seed(seed, model.t, "annual|W"))
    m_rng = np.random.default_rng(model_seed(seed, model.t, f"annual|{model.dep.name}"))
    rcfg = ocfg["rules"]
    history = history or {}
    Q = float(_percentile(history["m"].to_numpy(), np.array([m_bar]))[0]) if "m" in history else np.nan
    price_hist = history.get("prices")
    sma_n = int(ocfg["signals"]["sma_months"])
    lvl = np.ones((n_scenarios, len(tickers)))
    # ceny w sma_n − 1 datach przed t_j względem ceny w t_j (ostatni wiersz ``prices``)
    past = None
    if price_hist is not None and len(price_hist) >= sma_n and sma_n > 1:
        P = price_hist.reindex(columns=tickers).to_numpy(float)
        past = P[-sma_n:-1] / P[-1]
    path_prices: list[np.ndarray] = []
    out = {f"{sf}|{p}": np.ones(n_scenarios) for sf in sf_tickers for p in policies}
    for _ in range(months):
        sig_hat = state.avg_vol(n, td)[:, ti]
        sig = np.clip(m_bar * sig_hat, icfg["sigma_min"], icfg["sigma_max"])
        # sygnały P2 w ścieżce
        if past is not None:
            window = np.concatenate([np.broadcast_to(past[:, None, :], (len(past), n_scenarios, len(tickers))),
                                     np.stack(path_prices + [lvl])], axis=0)[-sma_n:]
            T = lvl / window.mean(axis=0) - 1.0
        else:
            T = np.full_like(lvl, np.nan)
        if "sigma_hat" in history:
            sh = history["sigma_hat"]
            V = np.column_stack([_percentile(sh[t].to_numpy() if t in sh else np.array([]), sig_hat[:, k])
                                 for k, t in enumerate(tickers)])
        else:
            V = np.full_like(lvl, np.nan)
        # blok n dni
        U = model.dep.simulate(n_scenarios * n, m_rng, W=w_rng.random((n_scenarios * n, model.dep.d)))
        U = full_uniforms(U.reshape(n_scenarios, n, model.dep.d), margins, model.dep.names)
        logret = state.step_block(U, eps_cap)
        g = np.exp(logret[:, ti])
        fx = np.exp(logret[:, pos[FX]]) if FX in pos else np.ones(n_scenarios)
        f = np.where(usd[None, :], fx[:, None], 1.0)
        G = g * f
        for pname, spec in policies.items():
            legs = policy_legs(pname, spec, tickers, Q, T, V, rcfg, frozen, n_scenarios)
            D = np.ones_like(G)
            Pi = np.zeros_like(G)
            for typ, m, h in legs:
                call = typ == "call"
                prem = bs.price(1.0, m, tau, r_i[None, :], 0.0, sig, call)
                if call:
                    D -= h * prem * (1 - kappa)
                    Pi -= h * np.maximum(g - m, 0.0)
                else:
                    D += h * prem * (1 + kappa)
                    Pi += h * np.maximum(m - g, 0.0)
            for sf, tks in sf_tickers.items():
                idx = [k for k, t in enumerate(tickers) if t in tks]
                if not idx:
                    continue
                R = (G[:, idx] + f[:, idx] * Pi[:, idx]).mean(axis=1) / D[:, idx].mean(axis=1)
                out[f"{sf}|{pname}"] *= R
        path_prices.append(lvl.copy())
        lvl = lvl * g
    return pd.DataFrame({k: v - 1.0 for k, v in out.items()})


def policy_legs(name: str, spec: dict, tickers: list[str], Q: float, T: np.ndarray, V: np.ndarray, rcfg: dict,
                frozen: pd.DataFrame | None, S: int) -> list[tuple[str, float, np.ndarray]]:
    """Nogi (typ, K/S, h (S × n)) polityki w ścieżkach."""
    nt = len(tickers)
    kind = spec.get("kind")
    if kind == "none" or name == "P0":
        return []
    if kind == "static":
        legs = []
        if "put" in spec:
            legs.append(("put", float(spec["put"]), np.full((S, nt), float(spec.get("h_put", 1.0)))))
        if "call" in spec:
            legs.append(("call", float(spec["call"]), np.full((S, nt), float(spec.get("h_call", 1.0)))))
        return legs
    if kind == "frozen":
        if frozen is None or frozen.empty:
            return []
        legs = []
        for (typ, m), grp in frozen.groupby(["type", "moneyness"]):
            h = grp.set_index("ticker")["h"].reindex(tickers).fillna(0.0).to_numpy(float)
            legs.append((typ, float(m), np.broadcast_to(h, (S, nt)).copy()))
        return legs
    if kind == "rules":
        qc, qe = rcfg["q_cheap"], rcfg["q_expensive"]
        with np.errstate(invalid="ignore"):
            put_cheap = np.full((S, nt), Q <= qc)
            put_trend = ~put_cheap & (T < 0) & (Q <= qe)
            collar = ~put_cheap & ~put_trend & (T < 0) & (Q > qe)
            cc = ~put_cheap & ~put_trend & ~collar & (Q >= qe) & (T > 0) & (V <= rcfg["v_max"])
        legs = [("put", float(rcfg["put_cheap"]), put_cheap.astype(float)),
                ("put", float(rcfg["put_trend"]), put_trend.astype(float)),
                ("put", float(rcfg["collar_put"]), collar.astype(float)),
                ("call", float(rcfg["collar_call"]), collar.astype(float)),
                ("call", float(rcfg["covered_call"]), cc.astype(float))]
        return _merge_legs(legs)
    raise ValueError(f"Polityka {name}: nieobsługiwany typ w ścieżkach ({kind})")


def _merge_legs(legs):
    merged: dict[tuple[str, float], np.ndarray] = {}
    for typ, m, h in legs:
        merged[(typ, m)] = merged.get((typ, m), 0.0) + h
    return [(typ, m, h) for (typ, m), h in merged.items() if np.any(h > 0)]


def summary(R: pd.DataFrame, alpha: float = 0.975) -> pd.DataFrame:
    """Statystyki rozkładu stóp w horyzoncie: średnia, mediana, odchylenie, VaR/ES_α strat, P(R < 0)."""
    from pf.opt.cvar import empirical_es

    rows = []
    for col in R.columns:
        x = R[col].to_numpy(float)
        sf, pol = col.split("|", 1)
        rows.append({"subfund": sf, "policy": pol, "mean": x.mean(), "median": float(np.median(x)), "sd": x.std(ddof=1),
                     "var": float(-np.quantile(x, 1 - alpha)), "es": empirical_es(-x, alpha),
                     "p_loss": float(np.mean(x < 0)), "q05": float(np.quantile(x, 0.05)),
                     "q95": float(np.quantile(x, 0.95))})
    return pd.DataFrame(rows)
