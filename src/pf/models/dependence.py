"""Wspólny interfejs modeli zależności (kopuł) i rejestr nazw.

Wszystkie modele estymowane są na tych samych pseudo-obserwacjach U ∈ (0, 1)^{n×d} (rangi
standaryzowanych reszt wspólnych marginesów AR(1)–GJR-GARCH(1,1) skew-t, estymowanych raz na datę).

| nazwa       | model                                                                                   |
|-------------|-----------------------------------------------------------------------------------------|
| rvine       | R-vine (Dißmann, obcięcie, kopuły parowe z AIC; ``pf.models.rvine``)                    |
| cvine       | C-vine – ta sama selekcja rodzin i obcięcie, w każdym drzewie korzeń max Σ|τ̂|          |
| dvine       | D-vine – porządek z maksymalnej ścieżki Hamiltona na |τ̂| (zachłanna + 2-opt)           |
| gauss       | kopuła Gaussa, R = corr(Φ⁻¹(U)), naprawa PSD (Higham)                                   |
| student_t   | kopuła t, R z ρᵢⱼ = sin(π τ̂ᵢⱼ / 2) (Higham), ν z profilu wiarygodności na siatce        |
| indep       | kopuła niezależności Π                                                                  |
| empirical   | kopuła empiryczna: losowanie wierszy panelu pseudo-obserwacji (bez gęstości)            |

Gęstości (x – wektor wyników Φ⁻¹(u) lub t_ν⁻¹(u), R = LLᵀ):
    Gauss: log c(u) = −½ ln|R| − ½ xᵀ(R⁻¹ − I)x;
    t:     log c(u) = ln Γ((ν+d)/2) + (d−1) ln Γ(ν/2) − d ln Γ((ν+1)/2) − ½ ln|R|
                      − (ν+d)/2 · ln(1 + xᵀR⁻¹x/ν) + (ν+1)/2 · Σᵢ ln(1 + xᵢ²/ν).
Symulacja: Gauss – U = Φ(Z Lᵀ), Z = Φ⁻¹(W); t – U = t_ν(Z Lᵀ / √(V/ν)), V ~ χ²_ν;
vine – algorytm z ``RVine.simulate``. Wejściowe W ~ U(0, 1)^{n×d} (``simulate(n, rng, W)``) pozwalają
użyć wspólnych liczb losowych we wszystkich modelach (vine, Gauss, niezależność; t – dodatkowo V z ``rng``;
kopuła empiryczna – numer wiersza ⌊W₁·n_obs⌋).

Kryteria: AIC = 2k − 2ℓ, BIC = k ln n − 2ℓ (ℓ – log-wiarygodność kopuły na oknie estymacji, k – liczba
parametrów; Gauss: d(d−1)/2, t: d(d−1)/2 + 1). Kopuła empiryczna nie ma gęstości (AIC/BIC = NaN).

Naprawa PSD (Higham 2002): naprzemienne rzuty na stożek macierzy dodatnio półokreślonych
(z poprawką Dykstry) i na zbiór macierzy o jedynkach na przekątnej; na końcu wartości własne ≥ ``eig_min``
i ponowna normalizacja do macierzy korelacji.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy import linalg, optimize, special

from pf.models.copulas import EPS
from pf.models.rvine import RVine, kendall_matrix

REGISTRY: dict[str, type["DependenceModel"]] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def _clip(U) -> np.ndarray:
    return np.clip(np.asarray(U, dtype=float), EPS, 1.0 - EPS)


def _dep_cfg(cfg: dict | None) -> dict:
    return (cfg or {}).get("dependence", {})


# --------------------------------------------------------------------------- macierze korelacji

def nearest_correlation(A: np.ndarray, eig_min: float = 1e-6, tol: float = 1e-9, max_iter: int = 500) -> np.ndarray:
    """Najbliższa (Frobenius) macierz korelacji – algorytm Highama (2002) z poprawką Dykstry."""
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    if np.linalg.eigvalsh(A).min() >= eig_min and np.allclose(np.diag(A), 1.0):
        return A
    Y, dS = A.copy(), np.zeros_like(A)
    for _ in range(max_iter):
        R = Y - dS
        w, V = np.linalg.eigh(R)
        X = (V * np.maximum(w, 0.0)) @ V.T
        dS = X - R
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)
        done = np.linalg.norm(Y_new - Y, "fro") / max(np.linalg.norm(Y_new, "fro"), 1e-300) < tol
        Y = Y_new
        if done:
            break
    w, V = np.linalg.eigh(0.5 * (Y + Y.T))
    Y = (V * np.maximum(w, eig_min)) @ V.T
    d = np.sqrt(np.diag(Y))
    Y = Y / np.outer(d, d)
    np.fill_diagonal(Y, 1.0)
    return 0.5 * (Y + Y.T)


def _cholesky(R: np.ndarray) -> np.ndarray:
    return linalg.cholesky(R, lower=True)


def _encode(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def _decode(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.float64).reshape(d["shape"]).copy()


# --------------------------------------------------------------------------- interfejs

class DependenceModel:
    """Model zależności: estymacja, symulacja, gęstość, kryteria informacyjne, serializacja."""

    name: ClassVar[str] = ""
    has_density: ClassVar[bool] = True

    def __init__(self, names: list[str], n_obs: int, loglik: float):
        self.names = list(names)
        self.n_obs = int(n_obs)
        self._loglik = float(loglik)

    @property
    def d(self) -> int:
        return len(self.names)

    @classmethod
    def fit(cls, U: np.ndarray, names: list[str], cfg: dict | None = None) -> "DependenceModel":
        raise NotImplementedError

    def simulate(self, n: int, rng: np.random.Generator | None = None, W: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def logpdf(self, U: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"Model {self.name} nie ma gęstości")

    @property
    def loglik(self) -> float:
        return self._loglik if self.has_density else float("nan")

    @property
    def n_params(self) -> int:
        return 0

    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.loglik if self.has_density else float("nan")

    def bic(self) -> float:
        return np.log(max(self.n_obs, 1)) * self.n_params - 2.0 * self.loglik if self.has_density else float("nan")

    def _base_dict(self) -> dict:
        return {"model": self.name, "names": self.names, "n_obs": self.n_obs, "loglik": self._loglik}

    def to_dict(self) -> dict:
        return self._base_dict()

    @classmethod
    def from_dict(cls, d: dict) -> "DependenceModel":
        raise NotImplementedError

    @staticmethod
    def _uniforms(n: int, d: int, rng: np.random.Generator | None, W: np.ndarray | None) -> np.ndarray:
        if W is not None:
            if W.shape != (n, d):
                raise ValueError(f"W ma wymiar {W.shape}, oczekiwano {(n, d)}")
            return np.asarray(W, dtype=float)
        return (rng or np.random.default_rng()).random((n, d))


# --------------------------------------------------------------------------- vine

class VineModel(DependenceModel):
    structure: ClassVar[str] = "rvine"

    def __init__(self, vine: RVine):
        super().__init__(vine.names, vine.n_obs, vine.loglik())
        self.vine = vine

    @classmethod
    def fit(cls, U, names, cfg=None):
        vcfg = (cfg or {}).get("rvine", {})
        vine = RVine.fit(np.asarray(U, dtype=float), names, truncation=vcfg.get("truncation", 3),
                         families=tuple(vcfg.get("families", ("gauss", "t", "clayton", "gumbel"))),
                         rotations=vcfg.get("rotations", True), criterion=vcfg.get("criterion", "aic"),
                         indep_level=vcfg.get("indep_test_level", 0.05),
                         t_df_bounds=tuple(vcfg.get("t_df_bounds", (2.05, 30.0))),
                         estimation=vcfg.get("estimation", "itau"), structure=cls.structure)
        return cls(vine)

    def simulate(self, n, rng=None, W=None):
        return self.vine.simulate(n, rng, W)

    def logpdf(self, U):
        return self.vine.logpdf(_clip(U))

    @property
    def n_params(self) -> int:
        return self.vine.n_params()

    def to_dict(self):
        return {**self._base_dict(), "vine": self.vine.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(RVine.from_dict(d["vine"]))


@register
class RVineModel(VineModel):
    name = "rvine"
    structure = "rvine"


@register
class CVineModel(VineModel):
    name = "cvine"
    structure = "cvine"


@register
class DVineModel(VineModel):
    name = "dvine"
    structure = "dvine"


# --------------------------------------------------------------------------- kopuły eliptyczne

@register
class GaussCopula(DependenceModel):
    name = "gauss"

    def __init__(self, names, R: np.ndarray, n_obs: int, loglik: float = float("nan")):
        super().__init__(names, n_obs, loglik)
        self.R = np.asarray(R, dtype=float)
        self.L = _cholesky(self.R)

    @classmethod
    def fit(cls, U, names, cfg=None):
        dc = _dep_cfg(cfg)
        X = special.ndtri(_clip(U))
        R = nearest_correlation(np.corrcoef(X, rowvar=False), dc.get("psd_eig_min", 1e-6),
                                dc.get("higham_tol", 1e-9), dc.get("higham_max_iter", 500))
        m = cls(names, R, len(U))
        m._loglik = float(m.logpdf(U).sum())
        return m

    def logpdf(self, U):
        X = special.ndtri(_clip(U))
        Y = linalg.solve_triangular(self.L, X.T, lower=True)
        return -np.log(np.diag(self.L)).sum() - 0.5 * ((Y * Y).sum(axis=0) - (X * X).sum(axis=1))

    def simulate(self, n, rng=None, W=None):
        Z = special.ndtri(_clip(self._uniforms(n, self.d, rng, W)))
        return _clip(special.ndtr(Z @ self.L.T))

    @property
    def n_params(self) -> int:
        return self.d * (self.d - 1) // 2

    def to_dict(self):
        return {**self._base_dict(), "R": _encode(self.R)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], _decode(d["R"]), d["n_obs"], d["loglik"])


def t_copula_logpdf(U: np.ndarray, L: np.ndarray, nu: float) -> np.ndarray:
    d = L.shape[0]
    X = special.stdtrit(nu, _clip(U))
    Y = linalg.solve_triangular(L, X.T, lower=True)
    q = (Y * Y).sum(axis=0)
    const = (special.gammaln((nu + d) / 2.0) + (d - 1) * special.gammaln(nu / 2.0)
             - d * special.gammaln((nu + 1.0) / 2.0) - np.log(np.diag(L)).sum())
    return const - (nu + d) / 2.0 * np.log1p(q / nu) + (nu + 1.0) / 2.0 * np.log1p(X * X / nu).sum(axis=1)


@register
class StudentTCopula(DependenceModel):
    name = "student_t"

    def __init__(self, names, R: np.ndarray, nu: float, n_obs: int, loglik: float = float("nan"),
                 profile: dict | None = None):
        super().__init__(names, n_obs, loglik)
        self.R = np.asarray(R, dtype=float)
        self.L = _cholesky(self.R)
        self.nu = float(nu)
        self.profile = profile or {}

    @classmethod
    def fit(cls, U, names, cfg=None):
        dc = _dep_cfg(cfg)
        tcfg = dc.get("student_t", {})
        tau = kendall_matrix(np.asarray(U, dtype=float))
        R = nearest_correlation(np.sin(np.pi * tau / 2.0), dc.get("psd_eig_min", 1e-6),
                                dc.get("higham_tol", 1e-9), dc.get("higham_max_iter", 500))
        L = _cholesky(R)
        grid = [float(x) for x in tcfg.get("df_grid", [2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50])]
        ll = {nu: float(t_copula_logpdf(U, L, nu).sum()) for nu in grid}
        k = int(np.argmax([ll[nu] for nu in grid]))
        nu_hat = grid[k]
        if tcfg.get("refine", True) and 0 < k < len(grid) - 1:
            res = optimize.minimize_scalar(lambda nu: -t_copula_logpdf(U, L, nu).sum(),
                                           bounds=(grid[k - 1], grid[k + 1]), method="bounded",
                                           options={"xatol": 1e-2})
            if -res.fun > ll[nu_hat]:
                nu_hat = float(res.x)
                ll[nu_hat] = float(-res.fun)
        return cls(names, R, nu_hat, len(U), ll[nu_hat], {str(k_): v for k_, v in ll.items()})

    def logpdf(self, U):
        return t_copula_logpdf(U, self.L, self.nu)

    def simulate(self, n, rng=None, W=None):
        rng = rng or np.random.default_rng()
        Z = special.ndtri(_clip(self._uniforms(n, self.d, rng, W))) @ self.L.T
        V = rng.chisquare(self.nu, size=(n, 1))
        return _clip(special.stdtr(self.nu, Z / np.sqrt(V / self.nu)))

    @property
    def n_params(self) -> int:
        return self.d * (self.d - 1) // 2 + 1

    def to_dict(self):
        return {**self._base_dict(), "R": _encode(self.R), "nu": self.nu, "profile": self.profile}

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], _decode(d["R"]), d["nu"], d["n_obs"], d["loglik"], d.get("profile"))


# --------------------------------------------------------------------------- niezależność, empiryczna

@register
class IndependenceCopula(DependenceModel):
    name = "indep"

    @classmethod
    def fit(cls, U, names, cfg=None):
        return cls(names, len(U), 0.0)

    def logpdf(self, U):
        return np.zeros(len(U))

    def simulate(self, n, rng=None, W=None):
        return _clip(self._uniforms(n, self.d, rng, W))

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], d["n_obs"], 0.0)


@register
class EmpiricalCopula(DependenceModel):
    name = "empirical"
    has_density = False

    def __init__(self, names, U: np.ndarray):
        super().__init__(names, len(U), float("nan"))
        self.U = np.asarray(U, dtype=float)

    @classmethod
    def fit(cls, U, names, cfg=None):
        return cls(names, _clip(U))

    def simulate(self, n, rng=None, W=None):
        if W is not None:
            idx = np.minimum((np.asarray(W)[:, 0] * self.n_obs).astype(np.int64), self.n_obs - 1)
        else:
            idx = (rng or np.random.default_rng()).integers(0, self.n_obs, size=n)
        return self.U[idx].copy()

    def to_dict(self):
        return {**self._base_dict(), "U": _encode(self.U)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], _decode(d["U"]))


# --------------------------------------------------------------------------- rejestr i zapis

def fit_model(name: str, U: np.ndarray, names: list[str], cfg: dict | None = None) -> DependenceModel:
    if name not in REGISTRY:
        raise KeyError(f"Nieznany model zależności: {name} (dostępne: {', '.join(REGISTRY)})")
    return REGISTRY[name].fit(np.asarray(U, dtype=float), list(names), cfg)


def from_dict(d: dict) -> DependenceModel:
    return REGISTRY[d["model"]].from_dict(d)


def save_model(model: DependenceModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(model.to_dict(), separators=(",", ":")).encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    return path


def load_model(path: Path) -> DependenceModel:
    raw = path.read_bytes()
    return from_dict(json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw))
