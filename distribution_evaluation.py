"""Testy zgodności rozkładów empirycznych dziennych logarytmicznych stóp zwrotu.

Hipoteza zerowa H0: stopy zwrotu danego instrumentu pochodzą z rodziny rozkładów F(·; θ)
o nieznanych parametrach θ, szacowanych metodą największej wiarygodności (MNW).
Badane rodziny (słownik ``DISTRIBUTIONS``): rozkład normalny N(μ, σ²) oraz rozkład
t-Studenta z parametrami położenia i skali t_ν(μ, s).

Testy:
    * χ² Pearsona oraz test G (ilorazu wiarygodności) – na k klasach jednakowo
      prawdopodobnych pod F(·; θ̂), k = ⌈2·n^{2/5}⌉ (reguła Moore'a). Liczba stopni swobody
      k − 1 − m (m – liczba szacowanych parametrów). Ponieważ θ̂ szacowane jest z danych
      niepogrupowanych, rozkład graniczny leży między χ²(k−1−m) a χ²(k−1)
      (Chernoff, Lehmann 1954) – podana p-wartość jest więc przybliżeniem liberalnym.
    * Statystyki oparte na dystrybuancie empirycznej, liczone dla u_(i) = F(x_(i); θ̂):
        Kołmogorowa–Smirnowa  D  = max(D⁺, D⁻),
        Kuipera               V  = D⁺ + D⁻,
        Cramera–von Misesa    W² = 1/(12n) + Σ (u_(i) − (2i−1)/(2n))²,
        Andersona–Darlinga    A² = −n − (1/n) Σ (2i−1)[ln u_(i) + ln(1 − u_(n+1−i))].
      Przy parametrach szacowanych z próby standardowe wartości krytyczne tych testów
      nie obowiązują (problem Lillieforsa), dlatego p-wartości wyznaczane są metodą
      parametrycznego bootstrapu: B prób z F(·; θ̂), ponowna estymacja θ i obliczenie
      statystyk; p = (1 + #{T*_b ≥ T}) / (B + 1).

Uwaga: notowania GPW zawierają wiele zerowych stóp zwrotu (wiązania), co narusza
założenie ciągłości rozkładu i dodatkowo sprzyja odrzuceniu H0.

Wejście:  data/processed/log_returns_<wariant>.csv  (z data_manipulation.py)
Wyjście:  data/processed/distribution_tests_<wariant>.csv          – format długi
          data/processed/distribution_tests_<wariant>_pvalues.csv  – p-wartości (ticker × test)
          data/processed/distribution_tests_<wariant>_summary.csv  – odsetek odrzuceń H0

Uruchomienie:
    python distribution_evaluation.py [--variants local pln] [--boot 999] [--boot-t 199]
"""

from __future__ import annotations

import argparse
import logging
import math
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"

ALPHA = 0.05
EPS = 1e-12
EDF_TESTS = ["KS", "Kuiper", "CvM", "AD"]
TESTS = ["Pearson_chi2", "G"] + EDF_TESTS

log = logging.getLogger("distribution_evaluation")


# --------------------------------------------------------------------------- statystyki EDF

def edf_statistics(u: np.ndarray) -> dict[str, np.ndarray]:
    """Statystyki EDF dla u = F(x; θ̂); działa wierszowo dla macierzy (B × n)."""
    u = np.sort(np.clip(np.atleast_2d(u), EPS, 1 - EPS), axis=1)
    n = u.shape[1]
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - u, axis=1)
    d_minus = np.max(u - (i - 1) / n, axis=1)
    w2 = 1 / (12 * n) + np.sum((u - (2 * i - 1) / (2 * n)) ** 2, axis=1)
    a2 = -n - np.sum((2 * i - 1) * (np.log(u) + np.log1p(-u[:, ::-1])), axis=1) / n
    return {"KS": np.maximum(d_plus, d_minus), "Kuiper": d_plus + d_minus, "CvM": w2, "AD": a2}


# --------------------------------------------------------------------------- rodziny rozkładów

class Normal:
    name = "normal"
    n_params = 2

    @staticmethod
    def fit(x: np.ndarray) -> tuple:
        return float(np.mean(x)), float(np.std(x))  # MNW: σ̂ z dzielnikiem n

    @staticmethod
    def cdf(x, params):
        mu, sigma = params
        return special.ndtr((x - mu) / sigma)

    @staticmethod
    def ppf(q, params):
        mu, sigma = params
        return mu + sigma * special.ndtri(q)

    @staticmethod
    def bootstrap_statistics(params, n, b, rng, **_) -> dict[str, np.ndarray]:
        # Statystyki EDF przy estymowanych (μ, σ) nie zależą od prawdziwych (μ, σ) – symulacja N(0, 1).
        z = rng.standard_normal((b, n))
        z = (z - z.mean(axis=1, keepdims=True)) / z.std(axis=1, keepdims=True)
        return edf_statistics(special.ndtr(z))

    @staticmethod
    def format_params(params) -> str:
        return f"mu={params[0]:.6g}; sigma={params[1]:.6g}"


class StudentT:
    name = "student_t"
    n_params = 3

    @staticmethod
    def _nll(theta, x):
        loc, log_scale, log_df = theta
        return -np.sum(stats.t.logpdf(x, df=np.exp(log_df), loc=loc, scale=np.exp(log_scale)))

    @classmethod
    def fit(cls, x: np.ndarray, start: tuple | None = None) -> tuple:
        if start is None:
            df0, loc0, scale0 = stats.t.fit(x)
        else:
            df0, loc0, scale0 = start[2], start[0], start[1]
        df0 = min(max(df0, 1.01), 500)
        res = optimize.minimize(cls._nll, x0=[loc0, np.log(scale0), np.log(df0)], args=(x,),
                                method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
        loc, log_scale, log_df = res.x
        return float(loc), float(np.exp(log_scale)), float(np.exp(log_df))

    @staticmethod
    def cdf(x, params):
        loc, scale, df = params
        return stats.t.cdf(x, df=df, loc=loc, scale=scale)

    @staticmethod
    def ppf(q, params):
        loc, scale, df = params
        return stats.t.ppf(q, df=df, loc=loc, scale=scale)

    @classmethod
    def bootstrap_statistics(cls, params, n, b, rng, **_) -> dict[str, np.ndarray]:
        loc, scale, df = params
        out = {k: np.empty(b) for k in EDF_TESTS}
        for j in range(b):
            xb = loc + scale * rng.standard_t(df, size=n)
            pb = cls.fit(xb, start=params)
            s = edf_statistics(cls.cdf(xb, pb))
            for k in EDF_TESTS:
                out[k][j] = s[k][0]
        return out

    @staticmethod
    def format_params(params) -> str:
        return f"loc={params[0]:.6g}; scale={params[1]:.6g}; df={params[2]:.4g}"


DISTRIBUTIONS = {d.name: d for d in (Normal, StudentT)}


# --------------------------------------------------------------------------- testy

def binned_tests(x: np.ndarray, dist, params) -> dict[str, tuple[float, float]]:
    n = len(x)
    k = math.ceil(2 * n ** 0.4)
    inner_edges = dist.ppf(np.arange(1, k) / k, params)
    observed = np.bincount(np.searchsorted(inner_edges, x, side="right"), minlength=k)
    expected = np.full(k, n / k)
    ddof = dist.n_params
    chi2 = stats.power_divergence(observed, expected, ddof=ddof, lambda_="pearson")
    g = stats.power_divergence(observed, expected, ddof=ddof, lambda_="log-likelihood")
    return {"Pearson_chi2": (float(chi2.statistic), float(chi2.pvalue)),
            "G": (float(g.statistic), float(g.pvalue))}


def evaluate_series(item: tuple[str, np.ndarray], dist_names: list[str], boot: dict[str, int],
                    seed: int) -> list[dict]:
    ticker, x = item
    rows = []
    for dist_name in dist_names:
        dist = DISTRIBUTIONS[dist_name]
        rng = np.random.default_rng([seed, zlib.crc32(ticker.encode()), len(dist_name)])
        params = dist.fit(x)
        base = {"ticker": ticker, "distribution": dist_name, "n_obs": len(x),
                "params": dist.format_params(params)}

        for test, (stat, p) in binned_tests(x, dist, params).items():
            rows.append({**base, "test": test, "statistic": stat, "p_value": p,
                         "p_value_method": "asymptotic chi2"})

        observed = {k: float(v[0]) for k, v in edf_statistics(dist.cdf(x, params)).items()}
        b = boot[dist_name]
        simulated = dist.bootstrap_statistics(params, len(x), b, rng)
        for test in EDF_TESTS:
            p = (1 + np.sum(simulated[test] >= observed[test])) / (b + 1)
            rows.append({**base, "test": test, "statistic": observed[test], "p_value": float(p),
                         "p_value_method": f"parametric bootstrap B={b}"})
    return rows


def run_variant(variant: str, dist_names: list[str], boot: dict[str, int], seed: int,
                jobs: int) -> None:
    path = PROCESSED_DIR / f"log_returns_{variant}.csv"
    if not path.exists():
        log.warning("Brak pliku %s – uruchom najpierw data_manipulation.py", path)
        return
    returns = pd.read_csv(path, index_col="Date", parse_dates=True)
    items = [(t, returns[t].dropna().to_numpy()) for t in returns.columns]

    worker = partial(evaluate_series, dist_names=dist_names, boot=boot, seed=seed)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, res in enumerate(pool.map(worker, items), start=1):
            rows.extend(res)
            log.info("[%s] %d/%d %s", variant, i, len(items), items[i - 1][0])

    results = pd.DataFrame(rows)
    results[f"reject_H0_{int(ALPHA * 100)}pct"] = results["p_value"] < ALPHA
    results.to_csv(PROCESSED_DIR / f"distribution_tests_{variant}.csv", index=False)

    pvalues = results.pivot_table(index="ticker", columns=["distribution", "test"],
                                  values="p_value", sort=False)
    pvalues = pvalues.reindex(columns=pd.MultiIndex.from_product([dist_names, TESTS]))
    pvalues.to_csv(PROCESSED_DIR / f"distribution_tests_{variant}_pvalues.csv")

    summary = (results.groupby(["distribution", "test"], sort=False)
               [f"reject_H0_{int(ALPHA * 100)}pct"].agg(["sum", "count", "mean"])
               .rename(columns={"sum": "n_rejected", "count": "n_tickers", "mean": "share_rejected"}))
    summary.to_csv(PROCESSED_DIR / f"distribution_tests_{variant}_summary.csv")
    log.info("[%s] odsetek odrzuceń H0 (α = %.2f):\n%s", variant, ALPHA,
             summary["share_rejected"].unstack().reindex(columns=TESTS).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variants", nargs="+", default=["local", "pln"])
    parser.add_argument("--distributions", nargs="+", default=list(DISTRIBUTIONS),
                        choices=list(DISTRIBUTIONS))
    parser.add_argument("--boot", type=int, default=999, help="B dla rozkładu normalnego")
    parser.add_argument("--boot-t", type=int, default=199, help="B dla rozkładu t-Studenta")
    parser.add_argument("--seed", type=int, default=20110101)
    parser.add_argument("--jobs", type=int, default=None, help="liczba procesów")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    boot = {"normal": args.boot, "student_t": args.boot_t}
    for variant in args.variants:
        run_variant(variant, args.distributions, boot, args.seed, args.jobs)


if __name__ == "__main__":
    main()
