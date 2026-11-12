"""Modelowa zmienność implikowana tickerów sub-funduszy na okres opcji (σ_imp odtwarzana, nie obserwowana).

Moduł uzupełnia ``pf.options.iv`` (odwracanie wzoru BSM dla kwotowań rynkowych): tu σ_imp jest
konstruowana z prognoz GARCH i indeksów zmienności Cboe.

1. Prognoza GARCH średniej zmienności na okres opcji (n sesji, opóźnienie L sesji od końca okna ≤ t_j − 1):

       σ̂_{i,t}(τ) = √[(252/n) · Σ_{h=L+1}^{L+n} E_t(σ²_{i,t+h})],

   dla AR(1)–GJR-GARCH(1,1) ze standaryzowanym skośnym t Hansena (η, λ):
       E_t σ²_{t+1} = ω + (α + γ·1[ε_t < 0]) ε_t² + β σ_t²   (znane w t),
       E_t σ²_{t+h} = ω + ρ · E_t σ²_{t+h−1},   ρ = α + β + γ·κ₋,   κ₋ = E[z² 1{z < 0}],
   gdzie κ₋ liczone jest kwadraturą z gęstości skośnego t (dla λ = 0: κ₋ = 1/2).

2. Mnożnik premii za ryzyko zmienności: m_t = (atm_adj · VIX_t/100) / σ̂_{SPX,t}(21) – stosunek zmienności ATM
   S&P 500 do prognozy GARCH (``atm_adj_in_m``; przy atm_adj = 1: m_t = VIX/σ̂_SPX).
   Ograniczenie górne (``m_cap``): m_t ≤ kwantyl historii m_s z okna estymacji (dane ≤ t − 1), co usuwa skoki m_t
   po spokojnym okresie przy podwyższonym VIX.

3. Źródło ``iv_source`` (``config/options.yaml: monthly_overlay.implied_vol``):
   * VIX / OVX / GVZ: σ_imp = atm_adj · indeks/100; dla VIX przy τ ≠ 30 dni interpolacja wariancji całkowitej
     σ²T między tenorami VIX9D (9 dni) i VIX (30 dni) (``pf.options.surface.interp_total_variance``,
     płaska ekstrapolacja σ); brak notowania indeksu → źródło zastępcze ``proxy``;
   * proxy: σ_imp = clip(m_t · σ̂_{i,t}(τ), σ_min, σ_max).

4. Opcjonalna liniowa korekta skośności: σ_imp(K) = max(σ_imp + β·ln(K/F), σ_min) (domyślnie β = 0).

Założenie: przeniesienie premii za ryzyko zmienności z rynku USA (VIX vs GARCH S&P 500) na pozostałe
tickery, w tym spółki GPW i ETF-y obligacyjne.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import integrate, special

from pf.models.margins import MarginFit
from pf.options.surface import interp_total_variance

INDEX_SOURCES = ("VIX", "OVX", "GVZ")


# --------------------------------------------------------------------------- skośny t Hansena

def skewt_pdf(z, eta: float, lam: float) -> np.ndarray:
    """Gęstość standaryzowanego skośnego t Hansena (1994), parametryzacja ``arch.univariate.SkewStudent``."""
    z = np.asarray(z, dtype=float)
    c = np.exp(special.gammaln((eta + 1) / 2) - special.gammaln(eta / 2)) / np.sqrt(np.pi * (eta - 2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    s = np.where(z < -a / b, 1 - lam, 1 + lam)
    return b * c * (1 + ((b * z + a) / s) ** 2 / (eta - 2)) ** (-(eta + 1) / 2)


@lru_cache(maxsize=4096)
def _neg_second_moment(eta: float, lam: float) -> float:
    c = np.exp(special.gammaln((eta + 1) / 2) - special.gammaln(eta / 2)) / np.sqrt(np.pi * (eta - 2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    f = lambda z: z * z * float(skewt_pdf(z, eta, lam))  # noqa: E731
    mode = -a / b
    if mode < 0:
        v1 = integrate.quad(f, -np.inf, mode, limit=200)[0]
        v2 = integrate.quad(f, mode, 0.0, limit=200)[0]
        return float(v1 + v2)
    return float(integrate.quad(f, -np.inf, 0.0, limit=200)[0])


def neg_second_moment(eta: float, lam: float) -> float:
    """κ₋ = E[z² 1{z < 0}] dla standaryzowanego skośnego t (E z² = 1)."""
    if not (np.isfinite(eta) and np.isfinite(lam)):
        return 0.5
    return _neg_second_moment(round(float(eta), 10), round(float(lam), 10))


# --------------------------------------------------------------------------- prognoza GARCH

def persistence(fit: MarginFit) -> float:
    """ρ = α + β + γ·κ₋ (trwałość wariancji warunkowej GJR-GARCH)."""
    return fit.alpha + fit.beta + fit.gamma * neg_second_moment(fit.eta, fit.lam)


def variance_path(sigma2_next, omega, rho, n: int, skip: int = 0) -> np.ndarray:
    """E σ²_{T+h}, h = skip+1, …, skip+n, ze znanego σ²_{T+1} (wektorowo: tablice o zgodnych kształtach).

    Zwraca tablicę (…, n) w jednostkach σ²_{T+1}."""
    s = np.asarray(sigma2_next, dtype=float)
    omega, rho = np.asarray(omega, dtype=float), np.asarray(rho, dtype=float)
    out = []
    for h in range(1, skip + n + 1):
        if h > 1:
            s = omega + rho * s
        if h > skip:
            out.append(s)
    return np.stack(out, axis=-1)


def garch_vol(fit: MarginFit, n: int, skip: int = 0, trading_days: int = 252) -> float:
    """σ̂(τ) = √[(252/n) Σ_{h=L+1}^{L+n} E σ²_{T+h}] w ujęciu rocznym (ułamek)."""
    if fit.deterministic or n <= 0:
        return 0.0
    v = variance_path(fit.next_sigma2(), fit.omega, persistence(fit), n, skip)
    return float(np.sqrt(trading_days / n * v.sum()) / fit.scale)


def vrp_multiplier(vix_pct: float, sigma_spx: float) -> float:
    """m_t = (VIX_t/100) / σ̂_SPX,t(21)."""
    if not (np.isfinite(vix_pct) and np.isfinite(sigma_spx)) or sigma_spx <= 0:
        return float("nan")
    return float(vix_pct / 100.0 / sigma_spx)


def market_multiplier(vix_pct: float, sigma_spx: float, icfg: dict) -> float:
    """m_t z korektą VIX → σ_ATM: (atm_adj · VIX_t/100) / σ̂_SPX,t(21), gdy ``atm_adj_in_m`` (domyślnie tak).

    VIX przewyższa zmienność ATM S&P 500; bez korekty m_t przenosiłby tę różnicę na σ_imp tickerów proxy."""
    adj = float(icfg.get("atm_adj", 1.0)) if icfg.get("atm_adj_in_m", True) else 1.0
    return vrp_multiplier(adj * vix_pct, sigma_spx)


def filtered_variance(fit: MarginFit, r) -> tuple[np.ndarray, np.ndarray]:
    """Innowacje ε_s i wariancje σ²_s (×scale²) z filtru AR(1)–GJR-GARCH przy parametrach ``fit`` na log-stopach ``r``.

    Start filtru: wariancja próbkowa innowacji (wpływ startu zanika geometrycznie w tempie β)."""
    y = fit.scale * np.asarray(r, dtype=float)
    eps = y[1:] - fit.mu - fit.phi * y[:-1]
    s2 = np.empty(len(eps))
    if len(eps):
        s2[0] = float(np.var(eps))
    for i in range(1, len(eps)):
        lev = fit.gamma if eps[i - 1] < 0 else 0.0
        s2[i] = fit.omega + (fit.alpha + lev) * eps[i - 1] ** 2 + fit.beta * s2[i - 1]
    return eps, s2


def multiplier_history(fit: MarginFit, r: pd.Series, vix: pd.Series, n: int, skip: int = 0, trading_days: int = 252,
                       atm_adj: float = 1.0) -> pd.Series:
    """Historia m_s w oknie estymacji: (atm_adj · VIX_{s+1}/100) / σ̂_{SPX,s}(n) dla sesji s okna.

    σ̂_{SPX,s}(n) to prognoza z informacji ≤ s przy parametrach ``fit``, a VIX pochodzi z kolejnej sesji okna, tak jak
    w m_t (model z danych ≤ t − 1, VIX z t). Ostatnia sesja okna nie ma pary, więc szereg używa tylko danych z okna.
    ``r`` – log-stopy S&P 500 w oknie (indeks dat), ``vix`` – poziomy VIX [p.p.] na tych samych datach."""
    eps, s2 = filtered_variance(fit, r.to_numpy(float))
    if len(eps) < 2:
        return pd.Series(dtype=float)
    nxt = fit.omega + (fit.alpha + np.where(eps < 0, fit.gamma, 0.0)) * eps ** 2 + fit.beta * s2
    v = variance_path(nxt, fit.omega, persistence(fit), n, skip)
    sig = np.sqrt(trading_days / n * v.sum(axis=-1)) / fit.scale
    lvl = vix.reindex(r.index).to_numpy(float)[2:]          # VIX w sesji s + 1; eps[0] odpowiada r.index[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(sig[:-1] > 0, atm_adj * lvl / 100.0 / sig[:-1], np.nan)
    return pd.Series(m, index=r.index[2:], name="m")


def multiplier_cap(history, ccfg: dict | None) -> float:
    """Górne ograniczenie m_t: kwantyl ``quantile`` historii m_s z okna estymacji (``m_cap`` w konfiguracji).

    Brak sekcji, ``quantile: null`` albo mniej niż ``min_obs`` skończonych obserwacji → +inf (bez ograniczenia)."""
    q = (ccfg or {}).get("quantile")
    if q is None:
        return float("inf")
    h = np.asarray(history, dtype=float)
    h = h[np.isfinite(h)]
    if len(h) < int((ccfg or {}).get("min_obs", 250)):
        return float("inf")
    return float(np.quantile(h, float(q)))


def capped_multiplier(m: float, cap: float) -> float:
    """min(m_t, cap); NaN przechodzi bez zmian."""
    return float(min(m, cap)) if np.isfinite(m) else m


# --------------------------------------------------------------------------- indeksy zmienności

def index_vol(source: str, levels: dict[str, float], days: float, vix_tenors_days: dict[str, int],
              atm_adj: float = 1.0) -> float:
    """σ_ATM z indeksu Cboe (ułamek); VIX: interpolacja wariancji całkowitej VIX9D–VIX po T = days/365."""
    if source == "VIX":
        cols = [c for c in vix_tenors_days if np.isfinite(levels.get(c, np.nan))]
        if "VIX" not in cols:
            return float("nan")
        tenors = np.array([vix_tenors_days[c] for c in cols]) / 365.0
        vols = np.array([levels[c] for c in cols]) / 100.0
        return float(atm_adj * interp_total_variance(tenors, vols, np.array(days / 365.0)))
    lv = levels.get(source, np.nan)
    return float(atm_adj * lv / 100.0) if np.isfinite(lv) else float("nan")


def proxy_vol(m: float, sigma_hat, sigma_min: float, sigma_max: float) -> np.ndarray:
    return np.clip(m * np.asarray(sigma_hat, dtype=float), sigma_min, sigma_max)


def strike_vol(sigma, K, F, beta: float = 0.0, sigma_min: float = 0.01) -> np.ndarray:
    """σ_imp(K) = max(σ_imp + β·ln(K/F), σ_min)."""
    sigma = np.asarray(sigma, dtype=float)
    if beta == 0.0:
        return sigma
    return np.maximum(sigma + beta * np.log(np.asarray(K, dtype=float) / np.asarray(F, dtype=float)), sigma_min)


def iv_source(ticker: str, icfg: dict) -> str:
    return icfg.get("sources", {}).get(ticker, icfg.get("default_source", "proxy"))


def sigma_table(tickers: list[str], fits: dict[str, MarginFit], n: int, skip: int, m: float,
                levels: dict[str, float], days: float, icfg: dict, trading_days: int = 252,
                shift_pp: float = 0.0) -> pd.DataFrame:
    """σ̂, źródło i σ_imp (ATM, przed korektą skośności) dla tickerów w dacie t_j.

    ``shift_pp`` – przesunięcie σ_imp tickerów ``proxy`` o ± p.p. (analiza wrażliwości)."""
    rows = []
    for t in tickers:
        sh = garch_vol(fits[t], n, skip, trading_days) if t in fits else float("nan")
        src = iv_source(t, icfg)
        sig = float("nan")
        eff = src
        if src in INDEX_SOURCES:
            sig = index_vol(src, levels, days, icfg["vix_tenors_days"], icfg.get("atm_adj", 1.0))
            if np.isfinite(sig):
                sig = float(np.clip(sig, icfg["sigma_min"], icfg["sigma_max"]))
        if not np.isfinite(sig):
            eff = "proxy"
            sig = float(proxy_vol(m, sh, icfg["sigma_min"], icfg["sigma_max"])) if np.isfinite(m) else float("nan")
            if np.isfinite(sig) and shift_pp:
                sig = float(np.clip(sig + shift_pp / 100.0, icfg["sigma_min"], icfg["sigma_max"]))
        rows.append({"ticker": t, "sigma_hat": sh, "iv_source_cfg": src, "iv_source": eff, "sigma_imp": sig})
    return pd.DataFrame(rows).set_index("ticker")


# --------------------------------------------------------------------------- kalibracja atm_adj

def calibrate_atm_adj(obs: pd.DataFrame, max_abs_k: float = 0.02, days: tuple[int, int] = (20, 45)) -> dict:
    """atm_adj ≈ mediana σ_ATM,obs / σ_VIX(T) dla opcji bliskich ATM (|k| ≤ ``max_abs_k``) o terminie w ``days``.

    ``obs`` – wynik ``pf.options.surface.prepare_quotes`` (kolumny iv, k, vix_T, T, reject)."""
    d = obs[(obs.get("reject", "") == "") & obs["iv"].notna() & obs["vix_T"].notna()]
    d = d[(d["k"].abs() <= max_abs_k) & (d["T"] * 365 >= days[0]) & (d["T"] * 365 <= days[1])]
    if d.empty:
        return {"status": "no_data", "n_obs": 0}
    ratio = d["iv"] / d["vix_T"]
    return {"status": "ok", "atm_adj": float(ratio.median()), "n_obs": int(len(d)),
            "iqr": [float(ratio.quantile(0.25)), float(ratio.quantile(0.75))],
            "first_date": str(pd.Timestamp(d["quote_date"].min()).date()) if "quote_date" in d else None,
            "last_date": str(pd.Timestamp(d["quote_date"].max()).date()) if "quote_date" in d else None}
