"""MC ex post ryzyka zrealizowanych szeregów portfeli: miesięczne VaR_α i ES_α z błędem standardowym.

Dane: dzienne stopy netto szeregu (portfel lub SF) w próbie (kalibracja, hold-out, całość).
Horyzont H = 21 sesji, stopa miesięczna R = Π_h (1 + r_h) − 1, strata L = −R.

1. FHS (filtered historical simulation, Barone-Adesi i in.): GJR-GARCH(1,1) ze skośnym t na 100·r_t
   (stała średnia), reszty standaryzowane ẑ_t; ścieżka: stan startowy (σ², ε) z losowej daty historycznej
   (rozkład bezwarunkowy stanu), dalej σ²_{h} = ω + (α + γ·1[ε_{h−1} < 0]) ε²_{h−1} + β σ²_{h−1},
   ε_h = σ_h ẑ*, ẑ* losowane ze zwracaniem z {ẑ_t}.
2. Bootstrap stacjonarny Politisa–Romano (``arch.bootstrap.StationaryBootstrap``): bloki o geometrycznej
   długości ze średnią b, b = max(b̂(r), b̂(r²)) z ``optimal_block_length`` (b̂(r²) zachowuje skupianie
   zmienności); każda replikacja długości T daje ⌊T/H⌋ kolejnych, rozłącznych ścieżek 21-dniowych
   (sąsiednie ścieżki są słabo zależne – SE jest wtedy nieco zaniżony).
3. Wielowymiarowe MC (strategia główna): marginesy AR(1)–GJR-GARCH(1,1) skew-t na dziennych log-stopach
   sub-funduszy, R-vine na resztach, stan startowy wszystkich marginesów z tej samej losowej daty;
   portfel R = Σ_k W_k R_k (kupno i trzymanie SF w miesiącu) dla W z ostatniej daty i średnich W okresu.

Liczba ścieżek N dobierana adaptacyjnie (partie, stop przy SE(ÊS)/|ÊS| ≤ ε lub N ≥ n_max), SE ze wzoru
z ``pf.sim.scenarios.es_with_se``. Szereg strategii o zmiennych w czasie wagach nie jest ściśle
stacjonarny – wyniki opisują rozkład bezwarunkowy „przeciętnego” miesiąca próby.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pf.sim.scenarios import es_with_se


def adaptive(draw, alpha: float, batch: int, n_min: int, n_max: int, rel_se: float) -> dict:
    """Losowanie partii strat ``draw(b)`` do SE/|ES| ≤ rel_se lub n_max."""
    parts, N = [], 0
    while N < n_max:
        b = min(batch, n_max - N)
        parts.append(np.asarray(draw(b), dtype=float))
        N += len(parts[-1])
        if N >= n_min:
            var, es, se = es_with_se(np.concatenate(parts), alpha)
            if abs(es) > 0 and se / abs(es) <= rel_se:
                return {"n": N, "var": var, "es": es, "es_se": se, "converged": True}
    var, es, se = es_with_se(np.concatenate(parts), alpha)
    return {"n": N, "var": var, "es": es, "es_se": se, "converged": bool(abs(es) > 0 and se / abs(es) <= rel_se)}


# --------------------------------------------------------------------------- FHS

class FHS:
    """GJR-GARCH(1,1) skew-t na stopach dziennych (×100) i symulacja ścieżek H-dniowych."""

    def __init__(self, r: np.ndarray, scale: float = 100.0):
        from arch import arch_model

        y = scale * np.asarray(r, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewt",
                             rescale=False).fit(disp="off", show_warning=False)
        p = res.params
        self.mu, self.omega = float(p["mu"]), float(p["omega"])
        self.alpha, self.gamma, self.beta = float(p["alpha[1]"]), float(p["gamma[1]"]), float(p["beta[1]"])
        self.z = np.asarray(res.std_resid, dtype=float)
        self.z = self.z[np.isfinite(self.z)]
        self.sigma2 = np.asarray(res.conditional_volatility, dtype=float) ** 2
        self.eps = np.asarray(res.resid, dtype=float)
        self.scale = scale
        self.converged = bool(res.convergence_flag == 0)

    def monthly_losses(self, n: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
        start = rng.integers(0, len(self.sigma2), n)
        s2 = self.omega + (self.alpha + self.gamma * (self.eps[start] < 0)) * self.eps[start] ** 2 \
            + self.beta * self.sigma2[start]
        growth = np.ones(n)
        for _ in range(horizon):
            e = np.sqrt(s2) * self.z[rng.integers(0, len(self.z), n)]
            growth *= 1.0 + (self.mu + e) / self.scale
            s2 = self.omega + (self.alpha + self.gamma * (e < 0)) * e ** 2 + self.beta * s2
        return -(growth - 1.0)


# --------------------------------------------------------------------------- bootstrap stacjonarny

def block_length(r: np.ndarray) -> float:
    from arch.bootstrap import optimal_block_length

    r = np.asarray(r, dtype=float)
    b = optimal_block_length(np.column_stack([r, r * r]))["stationary"].to_numpy()
    return float(max(1.0, np.nanmax(b)))


def stationary_monthly_losses(r: np.ndarray, n: int, horizon: int, block: float, seed: int) -> np.ndarray:
    """Straty miesięczne z replikacji bootstrapu stacjonarnego (⌊T/H⌋ rozłącznych ścieżek na replikację)."""
    from arch.bootstrap import StationaryBootstrap

    r = np.asarray(r, dtype=float)
    T = len(r)
    k = T // horizon
    bs = StationaryBootstrap(block, r, seed=int(seed))
    out = []
    got = 0
    while got < n:
        idx = bs.update_indices()[: k * horizon].reshape(k, horizon)
        out.append(-(np.prod(1.0 + r[idx], axis=1) - 1.0))
        got += k
    return np.concatenate(out)[:n]


# --------------------------------------------------------------------------- wielowymiarowe MC

class MultiMC:
    """Marginesy AR(1)–GJR-GARCH skew-t + R-vine na dziennych stopach SF; miesięczne stopy SF (S × K)."""

    def __init__(self, r_sf: pd.DataFrame, cfg: dict):
        from pf.models.dependence import fit_model
        from pf.models.margins import fit_margins
        from pf.models.rvine import pseudo_observations

        logr = np.log1p(r_sf)
        self.names = list(r_sf.columns)
        self.margins, z = fit_margins(logr, scale=cfg["margins"]["scale"], jobs=1)
        Z = z.dropna(how="any")
        self.model = fit_model("rvine", pseudo_observations(Z.to_numpy()), self.names, cfg)
        # stany historyczne marginesów (wspólna data startu)
        self.states = self._states(logr)

    def _states(self, logr: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for k, f in self.margins.items():
            y = f.scale * logr[k].to_numpy()
            n = len(y)
            s2, eps, rp = np.empty(n), np.empty(n), np.empty(n)
            s2_prev = f.omega / max(1e-6, 1.0 - f.alpha - 0.5 * f.gamma - f.beta)
            e_prev, r_prev = 0.0, 0.0
            for t in range(n):
                s2_t = f.omega + (f.alpha + (f.gamma if e_prev < 0 else 0.0)) * e_prev ** 2 + f.beta * s2_prev
                e_t = y[t] - f.mu - f.phi * r_prev
                s2[t], eps[t], rp[t] = s2_t, e_t, y[t]
                s2_prev, e_prev, r_prev = s2_t, e_t, y[t]
            out[k] = np.column_stack([s2, eps, rp])
        return out

    def monthly_returns(self, n: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
        from pf.models.margins import skewt_ppf

        K = len(self.names)
        T = len(next(iter(self.states.values())))
        start = rng.integers(0, T, n)
        U = self.model.simulate(n * horizon, rng).reshape(n, horizon, K)
        out = np.empty((n, K))
        for j, k in enumerate(self.names):
            f = self.margins[k]
            st = self.states[k][start]
            s2 = f.omega + (f.alpha + f.gamma * (st[:, 1] < 0)) * st[:, 1] ** 2 + f.beta * st[:, 0]
            r_prev = st[:, 2]
            Z = skewt_ppf(U[:, :, j].ravel(), f.eta, f.lam).reshape(n, horizon)
            total = np.zeros(n)
            for h in range(horizon):
                e = np.sqrt(s2) * Z[:, h]
                r = f.mu + f.phi * r_prev + e
                total += r
                s2 = f.omega + (f.alpha + f.gamma * (e < 0)) * e ** 2 + f.beta * s2
                r_prev = r
            out[:, j] = np.expm1(total / f.scale)
        return out
