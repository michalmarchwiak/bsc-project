"""Dwuwymiarowe kopuły parowe: Gaussa, t-Studenta, Claytona, Gumbela, Franka oraz obroty.

Dla kopuły C(u, v) z gęstością c(u, v) potrzebne są:
    h(u | v)  = ∂C(u, v)/∂v          – warunkowa dystrybuanta U przy V = v (h-funkcja),
    h⁻¹(w | v)                       – jej odwrotność względem u (symulacja).

Rodziny bazowe (wszystkie wymienne, C(u, v) = C(v, u)):
    Gauss:   ρ = sin(πτ/2);   h(u|v) = Φ((x − ρy)/√(1 − ρ²)),  x = Φ⁻¹(u), y = Φ⁻¹(v);
    t:       ρ = sin(πτ/2), ν z MNW przy ustalonym ρ;
             h(u|v) = t_{ν+1}((x − ρy) / √((ν + y²)(1 − ρ²)/(ν + 1))),  x = t_ν⁻¹(u), y = t_ν⁻¹(v);
    Clayton: θ = 2τ/(1 − τ);  C = (u^{−θ} + v^{−θ} − 1)^{−1/θ}, zależność w dolnym ogonie λ_L = 2^{−1/θ};
    Gumbel:  θ = 1/(1 − τ);   C = exp(−[(−ln u)^θ + (−ln v)^θ]^{1/θ}), górny ogon λ_U = 2 − 2^{1/θ};
             h⁻¹: dla x = −ln u, y = −ln v, A = (x^θ + y^θ)^{1/θ} równanie h(u|v) = w ma postać
             A + (θ − 1) ln A = −ln w − ln v + (θ − 1) ln y (metoda Newtona względem ln A, funkcja
             wypukła – zbieżność monotoniczna od prawej), x = (A^θ − y^θ)^{1/θ};
    Frank:   θ ∈ ℝ∖{0}, τ = 1 − 4/θ · [1 − D₁(θ)],  D₁(θ) = (1/θ)∫₀^θ t/(eᵗ − 1) dt (funkcja Debye'a);
             C = −(1/θ) ln(1 + A·C_v/B),  A = e^{−θu} − 1, C_v = e^{−θv} − 1, B = e^{−θ} − 1;
             c(u, v) = −θ B e^{−θ(u+v)} / (B + A C_v)²,  h(u|v) = e^{−θv} A / (B + A C_v),
             h⁻¹(w|v) = −(1/θ) ln(1 + w B / (e^{−θv} − w C_v));  brak zależności w ogonach (λ_L = λ_U = 0),
             symetria radialna; τ < 0 obsługiwane znakiem θ (bez obrotów).

Obroty o kąt r (Clayton, Gumbel), C₀ – kopuła bazowa:
    r = 180: c(u, v) = c₀(1−u, 1−v),  h(u|v) = 1 − h₀(1−u | 1−v)   (ogon przeciwny),
    r = 90:  c(u, v) = c₀(1−u, v),    h(u|v) = 1 − h₀(1−u | v)     (τ < 0),
    r = 270: c(u, v) = c₀(u, 1−v),    h(u|v) = h₀(u | 1−v)          (τ < 0).
Zamiana argumentów: kopuła (V, U) dla obrotu 90 to obrót 270 (i odwrotnie).

Estymacja (``estimation``): ``itau`` – parametry z odwrócenia τ Kendalla (ν kopuły t z MNW przy
ustalonym ρ); ``mle`` – MNW wszystkich parametrów kandydata (L-BFGS-B z ograniczeniami) startująca
z rozwiązania ``itau``. Rodzina wybierana kryterium AIC/BIC w punkcie estymatora.
Obliczenia prowadzone są w logarytmach, co zapewnia stabilność przy silnej zależności
(np. ETF-y na to samo aktywo, τ ≈ 0,9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import integrate, optimize, special, stats

EPS = 1e-12
RHO_MAX = 0.9999
CLAYTON_MAX, GUMBEL_MAX, FRANK_MAX = 50.0, 30.0, 80.0
FRANK_MIN = 1e-4                     # |θ| Franka (θ → 0: niezależność)
FAMILIES = ("indep", "gauss", "t", "clayton", "gumbel", "frank")
ROTATING = ("clayton", "gumbel")


def _clip(u) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)


# --------------------------------------------------------------------------- rodziny bazowe

def _gauss_logpdf(u, v, rho):
    x, y = special.ndtri(u), special.ndtri(v)
    r2 = 1.0 - rho * rho
    return -0.5 * np.log(r2) - (rho * rho * (x * x + y * y) - 2.0 * rho * x * y) / (2.0 * r2)


def _gauss_h(u, v, rho):
    return special.ndtr((special.ndtri(u) - rho * special.ndtri(v)) / np.sqrt(1.0 - rho * rho))


def _gauss_hinv(w, v, rho):
    return special.ndtr(special.ndtri(w) * np.sqrt(1.0 - rho * rho) + rho * special.ndtri(v))


def _t_logpdf(u, v, rho, nu):
    x, y = special.stdtrit(nu, u), special.stdtrit(nu, v)
    r2 = 1.0 - rho * rho
    const = (special.gammaln((nu + 2) / 2) + special.gammaln(nu / 2) - 2 * special.gammaln((nu + 1) / 2)
             - 0.5 * np.log(r2))
    quad = (x * x + y * y - 2.0 * rho * x * y) / (nu * r2)
    return (const - (nu + 2) / 2 * np.log1p(quad)
            + (nu + 1) / 2 * (np.log1p(x * x / nu) + np.log1p(y * y / nu)))


def _t_h(u, v, rho, nu):
    x, y = special.stdtrit(nu, u), special.stdtrit(nu, v)
    scale = np.sqrt((nu + y * y) * (1.0 - rho * rho) / (nu + 1.0))
    return special.stdtr(nu + 1.0, (x - rho * y) / scale)


def _t_hinv(w, v, rho, nu):
    y = special.stdtrit(nu, v)
    scale = np.sqrt((nu + y * y) * (1.0 - rho * rho) / (nu + 1.0))
    return special.stdtr(nu, special.stdtrit(nu + 1.0, w) * scale + rho * y)


def _clayton_logA(u, v, th):
    """ln A, A = u^{−θ} + v^{−θ} − 1, liczone bez przepełnienia: ln A = m + ln(1 + e^{−m}·expm1(min))."""
    a, b = -th * np.log(u), -th * np.log(v)
    m, n = np.maximum(a, b), np.minimum(a, b)
    return m + np.log1p(np.exp(-m) * np.expm1(n))


def _clayton_logpdf(u, v, th):
    return (np.log1p(th) - (1.0 + th) * (np.log(u) + np.log(v)) - (2.0 + 1.0 / th) * _clayton_logA(u, v, th))


def _clayton_h(u, v, th):
    return np.exp(-(1.0 + th) * np.log(v) - (1.0 + 1.0 / th) * _clayton_logA(u, v, th))


def _clayton_hinv(w, v, th):
    # A = (w v^{θ+1})^{−θ/(1+θ)};  u^{−θ} = 1 + e^{b}·expm1(ln A − b),  b = −θ ln v.
    log_a = -(np.log(w) + (th + 1.0) * np.log(v)) * th / (1.0 + th)
    b = -th * np.log(v)
    d = np.maximum(log_a - b, 1e-300)
    log_u_neg_th = np.logaddexp(0.0, b + np.log(np.expm1(d)))
    return np.exp(-log_u_neg_th / th)


def _gumbel_parts(u, v, th):
    lx, ly = np.log(-np.log(u)), np.log(-np.log(v))
    log_s = np.logaddexp(th * lx, th * ly)          # S = x^θ + y^θ
    return lx, ly, log_s, np.exp(log_s / th)         # A = S^{1/θ}


def _gumbel_logpdf(u, v, th):
    lx, ly, log_s, a = _gumbel_parts(u, v, th)
    return (-a - np.log(u) - np.log(v) + (th - 1.0) * (lx + ly) + (2.0 / th - 2.0) * log_s
            + np.log1p((th - 1.0) / a))


def _gumbel_h(u, v, th):
    _, ly, log_s, a = _gumbel_parts(u, v, th)
    return np.exp(-a - np.log(v) + (th - 1.0) * ly + (1.0 / th - 1.0) * log_s)


def _gumbel_hinv(w, v, th, max_iter: int = 80, tol: float = 1e-13):
    y = -np.log(v)
    ly = np.log(y)
    c = -np.log(w) - np.log(v) + (th - 1.0) * ly          # A + (θ − 1) ln A = c
    a = np.log(np.maximum(np.maximum(c, 1.0), y))           # g(a₀) ≥ 0: start po prawej stronie pierwiastka
    for _ in range(max_iter):
        ea = np.exp(a)
        step = (ea + (th - 1.0) * a - c) / (ea + (th - 1.0))
        a = a - step
        if np.max(np.abs(step)) < tol:
            break
    # ln x = ln A + (1/θ) ln(1 − (y/A)^θ)
    ratio = np.exp(np.minimum(th * (ly - a), 0.0))
    with np.errstate(divide="ignore"):
        lx = a + np.log1p(-np.minimum(ratio, 1.0 - 1e-16)) / th
    return np.exp(-np.exp(lx))


def _frank_denominator(u, v, th):
    """B + A·C_v = e^{−θu}·C_v + e^{−θv}·(e^{−θ(1−v)} − 1) – oba składniki tego samego znaku (bez redukcji cyfr)."""
    return np.exp(-th * u) * np.expm1(-th * v) + np.exp(-th * v) * np.expm1(-th * (1.0 - v))


def _frank_logpdf(u, v, th):
    return np.log(th * -np.expm1(-th)) - th * (u + v) - 2.0 * np.log(np.abs(_frank_denominator(u, v, th)))


def _frank_h(u, v, th):
    return np.exp(-th * v) * np.expm1(-th * u) / _frank_denominator(u, v, th)


def _frank_hinv(w, v, th):
    # e^{−θu} = [(1 − w) e^{−θv} + w e^{−θ}] / [(1 − w) e^{−θv} + w]
    a = np.log1p(-w) - th * v
    lw = np.log(w)
    return -(np.logaddexp(a, lw - th) - np.logaddexp(a, lw)) / th


def debye1(x: float) -> float:
    """D₁(x) = (1/x) ∫₀^x t/(eᵗ − 1) dt."""
    if abs(x) < 1e-8:
        return 1.0 - x / 4.0
    val = integrate.quad(lambda t: t / np.expm1(t) if t != 0 else 1.0, 0.0, x, limit=200)[0]
    return val / x


def frank_tau(th: float) -> float:
    if abs(th) < 1e-8:
        return 0.0
    return 1.0 - 4.0 / th * (1.0 - debye1(th))


def _bisect_hinv(hfun, w, v, iters: int = 52):
    """Odwrotność h(·|v) bisekcją na skali logitowej u (h rosnące w u)."""
    lo = np.full(np.shape(w), -40.0)
    hi = np.full(np.shape(w), 40.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        less = hfun(_clip(special.expit(mid)), v) < w
        lo, hi = np.where(less, mid, lo), np.where(less, hi, mid)
    return special.expit(0.5 * (lo + hi))


BASE = {
    "gauss": (_gauss_logpdf, _gauss_h, _gauss_hinv),
    "t": (_t_logpdf, _t_h, _t_hinv),
    "clayton": (_clayton_logpdf, _clayton_h, _clayton_hinv),
    "gumbel": (_gumbel_logpdf, _gumbel_h, _gumbel_hinv),
    "frank": (_frank_logpdf, _frank_h, _frank_hinv),
}


# --------------------------------------------------------------------------- τ ↔ parametr

def param_from_tau(family: str, tau: float) -> tuple[float, ...]:
    """Odwrócenie τ Kendalla; dla obróconych Claytona/Gumbela używane jest |τ|."""
    if family == "indep":
        return ()
    if family in ("gauss", "t"):
        return (float(np.clip(np.sin(np.pi * tau / 2.0), -RHO_MAX, RHO_MAX)),)
    if family == "frank":
        t = min(abs(tau), frank_tau(FRANK_MAX))
        if t <= frank_tau(FRANK_MIN):
            return (float(np.copysign(FRANK_MIN, tau if tau != 0 else 1.0)),)
        th = optimize.brentq(lambda x: frank_tau(x) - t, FRANK_MIN, FRANK_MAX, xtol=1e-10)
        return (float(np.copysign(th, tau)),)
    t = min(abs(tau), 0.98)
    if family == "clayton":
        return (float(np.clip(2.0 * t / (1.0 - t), 1e-4, CLAYTON_MAX)),)
    if family == "gumbel":
        return (float(np.clip(1.0 / (1.0 - t), 1.0 + 1e-6, GUMBEL_MAX)),)
    raise ValueError(family)


def tau_of(family: str, rotation: int, params: tuple[float, ...]) -> float:
    if family == "indep":
        return 0.0
    if family in ("gauss", "t"):
        return float(2.0 / np.pi * np.arcsin(params[0]))
    if family == "frank":
        return frank_tau(params[0])
    th = params[0]
    tau = th / (th + 2.0) if family == "clayton" else 1.0 - 1.0 / th
    return -tau if rotation in (90, 270) else tau


# --------------------------------------------------------------------------- kopuła parowa

@dataclass(frozen=True)
class PairCopula:
    family: str = "indep"
    rotation: int = 0
    params: tuple[float, ...] = field(default_factory=tuple)

    @property
    def n_params(self) -> int:
        return len(self.params)

    def _flip(self, u, v):
        r = self.rotation
        if r == 0:
            return u, v
        if r == 180:
            return 1.0 - u, 1.0 - v
        if r == 90:
            return 1.0 - u, v
        return u, 1.0 - v  # 270

    def logpdf(self, u, v) -> np.ndarray:
        u, v = _clip(u), _clip(v)
        if self.family == "indep":
            return np.zeros(np.broadcast(u, v).shape)
        a, b = self._flip(u, v)
        return BASE[self.family][0](_clip(a), _clip(b), *self.params)

    def h(self, u, v) -> np.ndarray:
        """h(u | v) = P(U ≤ u | V = v)."""
        u, v = _clip(u), _clip(v)
        if self.family == "indep":
            return np.broadcast_to(u, np.broadcast(u, v).shape).copy()
        h0 = lambda a, b: BASE[self.family][1](_clip(a), _clip(b), *self.params)
        r = self.rotation
        if r == 0:
            out = h0(u, v)
        elif r == 180:
            out = 1.0 - h0(1.0 - u, 1.0 - v)
        elif r == 90:
            out = 1.0 - h0(1.0 - u, v)
        else:
            out = h0(u, 1.0 - v)
        return _clip(out)

    def hinv(self, w, v) -> np.ndarray:
        """u takie, że h(u | v) = w."""
        w, v = _clip(w), _clip(v)
        if self.family == "indep":
            return np.broadcast_to(w, np.broadcast(w, v).shape).copy()
        fam = BASE[self.family]
        if fam[2] is not None:
            inv0 = lambda a, b: fam[2](_clip(a), _clip(b), *self.params)
        else:
            inv0 = lambda a, b: _bisect_hinv(lambda x, y: fam[1](x, y, *self.params), _clip(a), _clip(b))
        r = self.rotation
        if r == 0:
            out = inv0(w, v)
        elif r == 180:
            out = 1.0 - inv0(1.0 - w, 1.0 - v)
        elif r == 90:
            out = 1.0 - inv0(1.0 - w, v)
        else:
            out = inv0(w, 1.0 - v)
        return _clip(out)

    def swap(self) -> "PairCopula":
        """Kopuła pary (V, U)."""
        rot = {90: 270, 270: 90}.get(self.rotation, self.rotation)
        return PairCopula(self.family, rot, self.params)

    def h1(self, u, v) -> np.ndarray:
        """h(v | u) = P(V ≤ v | U = u)."""
        return self.swap().h(v, u)

    def hinv1(self, w, u) -> np.ndarray:
        """v takie, że h(v | u) = w."""
        return self.swap().hinv(w, u)

    def tau(self) -> float:
        return tau_of(self.family, self.rotation, self.params)

    def tail_dependence(self) -> tuple[float, float]:
        """(λ_L, λ_U) – współczynniki zależności w ogonach (dla obrotów 90/270: 0, 0)."""
        f, p = self.family, self.params
        if f == "t":
            rho, nu = p
            lam = float(2.0 * special.stdtr(nu + 1.0, -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))))
            return lam, lam
        if f == "clayton" and self.rotation in (0, 180):
            lam = float(2.0 ** (-1.0 / p[0]))
            return (lam, 0.0) if self.rotation == 0 else (0.0, lam)
        if f == "gumbel" and self.rotation in (0, 180):
            lam = float(2.0 - 2.0 ** (1.0 / p[0]))
            return (0.0, lam) if self.rotation == 0 else (lam, 0.0)
        return 0.0, 0.0

    def to_dict(self) -> dict:
        return {"family": self.family, "rotation": self.rotation, "params": list(self.params)}

    @staticmethod
    def from_dict(d: dict) -> "PairCopula":
        return PairCopula(d["family"], int(d["rotation"]), tuple(float(x) for x in d["params"]))

    def label(self) -> str:
        if self.family in ROTATING and self.rotation:
            return f"{self.family}{self.rotation}"
        return self.family


# --------------------------------------------------------------------------- estymacja

def kendall_tau(u, v) -> tuple[float, float]:
    """(τ, p-value) – τ_b Kendalla i test niezależności (rozkład asymptotyczny)."""
    res = stats.kendalltau(u, v)
    tau = 0.0 if not np.isfinite(res.statistic) else float(res.statistic)
    return tau, float(res.pvalue) if np.isfinite(res.pvalue) else 1.0


def candidates(tau: float, families, rotations: bool = True) -> list[tuple[str, int]]:
    out = []
    for f in families:
        if f in ("gauss", "t", "frank"):
            out.append((f, 0))
        elif f in ROTATING:
            rots = ((0, 180) if tau >= 0 else (90, 270)) if rotations else ((0,) if tau >= 0 else ())
            out += [(f, r) for r in rots]
    return out


def fit_t_df(u, v, rho: float, bounds=(2.05, 30.0)) -> float:
    res = optimize.minimize_scalar(lambda nu: -np.sum(_t_logpdf(u, v, rho, nu)), bounds=bounds,
                                   method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def param_bounds(family: str, start: tuple[float, ...], t_df_bounds=(2.05, 30.0)) -> list[tuple[float, float]]:
    """Przedziały parametrów dla MNW (Frank: znak θ ustalony przez punkt startowy z τ)."""
    if family == "gauss":
        return [(-RHO_MAX, RHO_MAX)]
    if family == "t":
        return [(-RHO_MAX, RHO_MAX), tuple(t_df_bounds)]
    if family == "clayton":
        return [(1e-4, CLAYTON_MAX)]
    if family == "gumbel":
        return [(1.0 + 1e-6, GUMBEL_MAX)]
    if family == "frank":
        return [(FRANK_MIN, FRANK_MAX)] if start[0] > 0 else [(-FRANK_MAX, -FRANK_MIN)]
    raise ValueError(family)


def fit_mle(u, v, family: str, rotation: int, start: tuple[float, ...],
            t_df_bounds=(2.05, 30.0)) -> tuple[tuple[float, ...], float]:
    """MNW parametrów kopuły (family, rotation) startująca z ``start``; zwraca (parametry, log-wiarygodność)."""
    u, v = _clip(u), _clip(v)
    cop0 = PairCopula(family, rotation, tuple(start))
    a, b = (_clip(x) for x in cop0._flip(u, v))
    logpdf = BASE[family][0]
    bounds = param_bounds(family, start, t_df_bounds)
    x0 = np.clip(np.asarray(start, dtype=float), [lo for lo, _ in bounds], [hi for _, hi in bounds])

    def nll(p):
        val = -np.sum(logpdf(a, b, *p))
        return val if np.isfinite(val) else 1e300

    ll0 = -nll(x0)
    res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
    if not np.isfinite(res.fun) or -res.fun < ll0:
        return tuple(float(x) for x in x0), float(ll0)
    return tuple(float(x) for x in res.x), float(-res.fun)


def select(u, v, families=("gauss", "t", "clayton", "gumbel"), rotations: bool = True,
           criterion: str = "aic", indep_level: float | None = 0.05,
           t_df_bounds=(2.05, 30.0), estimation: str = "itau") -> tuple[PairCopula, dict]:
    """Wybór kopuły parowej: parametry z τ Kendalla (``itau``) lub MNW (``mle``), rodzina z min. AIC/BIC.

    Zwraca kopułę i słownik diagnostyczny (tau, p-value, loglik, kryterium).
    """
    if estimation not in ("itau", "mle"):
        raise ValueError(f"Nieznana metoda estymacji: {estimation}")
    u, v = _clip(u), _clip(v)
    n = len(u)
    tau, pval = kendall_tau(u, v)
    info = {"tau": tau, "pvalue": pval, "n": n}
    if indep_level is not None and pval > indep_level:
        return PairCopula(), {**info, "loglik": 0.0, "crit": 0.0}
    penalty = 2.0 if criterion == "aic" else np.log(n)
    best, best_crit, best_ll = PairCopula(), np.inf, 0.0
    for fam, rot in candidates(tau, families, rotations):
        p = param_from_tau(fam, tau)
        if fam == "t":
            p = (p[0], fit_t_df(u, v, p[0], t_df_bounds))
        if estimation == "mle":
            p, ll = fit_mle(u, v, fam, rot, p, t_df_bounds)
            cop = PairCopula(fam, rot, p)
        else:
            cop = PairCopula(fam, rot, p)
            ll = float(np.sum(cop.logpdf(u, v)))
        crit = penalty * cop.n_params - 2.0 * ll
        if np.isfinite(crit) and crit < best_crit:
            best, best_crit, best_ll = cop, crit, ll
    if not np.isfinite(best_crit):
        best_crit = 0.0
    return best, {**info, "loglik": best_ll, "crit": best_crit}
