"""Rozkłady brzegowe: AR(1)–GJR-GARCH(1,1) z rozkładem skośnym t Hansena (pakiet ``arch``).

    r_t = μ + φ r_{t−1} + ε_t,   ε_t = σ_t z_t,   z_t ~ skew-t(η, λ),
    σ_t² = ω + (α + γ·1[ε_{t−1} < 0]) ε_{t−1}² + β σ_{t−1}².

Estymacja na stopach ×``scale`` (stabilność numeryczna) na dniach sesyjnych danego aktywa.
Do estymacji kopuły służą standaryzowane reszty ẑ_t (pseudo-obserwacje z rang – podejście
półparametryczne), do symulacji – kwantyle dopasowanego rozkładu skew-t.

Panel reszt łączy GPW i giełdy USA: dni, w których cały rynek był zamknięty (święta), są
usuwane, a pojedyncze braki (dni bez obrotu danego instrumentu) wypełniane są ẑ = 0.
Zamknięcie GPW (17:00) i NYSE (22:00 czasu polskiego) nie są synchroniczne, co zaniża
dzienną zależność między rynkami.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MarginFit:
    name: str
    mu: float
    phi: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    eta: float
    lam: float
    scale: float
    last_r: float        # ostatnia stopa (×scale)
    last_eps: float      # ostatnia innowacja ε_T (×scale)
    last_sigma2: float   # σ_T² (×scale²)
    loglik: float
    n_obs: int
    converged: bool

    def next_sigma2(self) -> float:
        lev = self.gamma if self.last_eps < 0 else 0.0
        return self.omega + (self.alpha + lev) * self.last_eps ** 2 + self.beta * self.last_sigma2

    def to_dict(self) -> dict:
        return asdict(self)


def fit_margin(r: pd.Series, scale: float = 100.0) -> tuple[MarginFit, pd.Series]:
    """Dopasowanie modelu do szeregu log-stóp (bez braków); zwraca parametry i reszty ẑ_t."""
    from arch import arch_model

    y = scale * r.dropna()
    model = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off", show_warning=False)
    p = res.params
    phi_name = [k for k in p.index if k.endswith("[1]") and k not in ("alpha[1]", "gamma[1]", "beta[1]")][0]
    z = res.std_resid.dropna()
    fit = MarginFit(
        name=str(r.name), mu=float(p["Const"]), phi=float(p[phi_name]), omega=float(p["omega"]),
        alpha=float(p["alpha[1]"]), gamma=float(p["gamma[1]"]), beta=float(p["beta[1]"]),
        eta=float(p["eta"]), lam=float(p["lambda"]), scale=scale, last_r=float(y.iloc[-1]),
        last_eps=float(res.resid.iloc[-1]), last_sigma2=float(res.conditional_volatility.iloc[-1] ** 2),
        loglik=float(res.loglikelihood), n_obs=int(len(y)), converged=bool(res.convergence_flag == 0),
    )
    return fit, z.rename(r.name)


def _fit_one(args):
    r, scale = args
    return fit_margin(r, scale)


def fit_margins(returns: pd.DataFrame, scale: float = 100.0, jobs: int | None = None
                ) -> tuple[dict[str, MarginFit], pd.DataFrame]:
    """Dopasowanie marginesów dla wszystkich kolumn; zwraca parametry i panel reszt ẑ."""
    tasks = [(returns[c], scale) for c in returns.columns]
    if jobs == 1:
        out = [_fit_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            out = list(ex.map(_fit_one, tasks))
    fits = {f.name: f for f, _ in out}
    bad = [n for n, f in fits.items() if not f.converged]
    if bad:
        log.warning("Brak zbieżności estymacji GARCH: %s", ", ".join(bad))
    z = pd.concat([z for _, z in out], axis=1).reindex(columns=returns.columns)
    return fits, z


def residual_panel(z: pd.DataFrame, market_of=lambda t: "PL" if t.endswith(".WA") else "US") -> pd.DataFrame:
    """Usuwa dni zamknięcia całego rynku, pozostałe braki zastępuje ẑ = 0."""
    markets = pd.Series({c: market_of(c) for c in z.columns})
    open_all = np.ones(len(z), dtype=bool)
    for m in markets.unique():
        open_all &= z[markets.index[markets == m]].notna().any(axis=1).to_numpy()
    return z[open_all].fillna(0.0)


def skewt_ppf(u: np.ndarray, eta: float, lam: float) -> np.ndarray:
    from arch.univariate import SkewStudent

    return SkewStudent().ppf(np.asarray(u, dtype=float), np.array([eta, lam]))


def simulate_paths(fits: list[MarginFit], U: np.ndarray) -> np.ndarray:
    """Skumulowane log-stopy w horyzoncie H dla scenariuszy z kopuły.

    ``U`` – tablica (S, H, d) wartości kopuły dla kolejnych dni; zwraca (S, d) sumy log-stóp
    (w jednostkach pierwotnych, tj. /scale), startując ze stanu GARCH na koniec próby.
    """
    S, H, d = U.shape
    assert d == len(fits)
    par = {k: np.array([getattr(f, k) for f in fits]) for k in
           ("mu", "phi", "omega", "alpha", "gamma", "beta", "scale", "last_r")}
    Z = np.empty_like(U)
    for j, f in enumerate(fits):
        Z[:, :, j] = skewt_ppf(U[:, :, j].ravel(), f.eta, f.lam).reshape(S, H)
    sigma2 = np.tile(np.array([f.next_sigma2() for f in fits]), (S, 1))
    r_prev = np.tile(par["last_r"], (S, 1))
    total = np.zeros((S, d))
    for h in range(H):
        eps = np.sqrt(sigma2) * Z[:, h, :]
        r = par["mu"] + par["phi"] * r_prev + eps
        total += r
        sigma2 = par["omega"] + (par["alpha"] + par["gamma"] * (eps < 0)) * eps ** 2 + par["beta"] * sigma2
        r_prev = r
    return total / par["scale"]
