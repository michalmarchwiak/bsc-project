"""Zastępcza zmienność implikowana dla opcji na WIG20 (założenie modelowe).

Dla WIG20 nie istnieje publikowany indeks zmienności implikowanej, a opcje OW20 na GPW
mają niską płynność (szerokie spready, nieliczne transakcje), więc nie nadają się do
kalibracji powierzchni. Przyjmujemy, że premia za ryzyko zmienności (stosunek zmienności
implikowanej do warunkowej) jest taka sama jak dla S&P 500:

    σ_imp,WIG20,t = σ_GARCH,WIG20,t · (VIX_t / σ_GARCH,SPX,t),

gdzie σ_GARCH – warunkowe zmienności roczne z modeli AR(1)–GJR-GARCH(1,1), a VIX_t w ujęciu
dziesiętnym, ratio = σ_GARCH,WIG20 / σ_GARCH,SPX. Wzór daje odpowiednik VIX dla WIG20; tę samą
proporcję stosujemy do zmienności ATM z całej struktury terminowej (σ_ATM,WIG20(T) =
ratio · σ_ATM,SPX(T), σ_ATM,SPX(T) = a + b·VIX_T – z korektą VIX → ATM), a nachylenie
skośności ∂σ/∂k przejmujemy z powierzchni S&P 500:

    σ_imp,WIG20(k, T) = ratio · σ_ATM,SPX(T) + (c + d·VIX_T)·k + δ,

δ – przesunięcie wrażliwości (domyślnie 0, analiza wrażliwości δ = ±3 p.p.).
Wycena opcji na WIG20 nie jest walidowana na danych rynkowych.
"""

from __future__ import annotations

import numpy as np

from pf.options.surface import SkewParams

DEFAULT_SHIFT_PP = 3.0


def vol_ratio(sigma_garch_wig20, sigma_garch_spx) -> np.ndarray:
    """ratio = σ_GARCH,WIG20 / σ_GARCH,SPX – mnożnik przenoszący zmienność implikowaną S&P 500 na WIG20."""
    return np.asarray(sigma_garch_wig20, dtype=float) / np.asarray(sigma_garch_spx, dtype=float)


def atm_vol_30d(sigma_garch_wig20, sigma_garch_spx, vix) -> np.ndarray:
    """σ_imp,WIG20,t = σ_GARCH,WIG20,t · VIX_t / σ_GARCH,SPX,t (30 dni, dziesiętnie)."""
    return vol_ratio(sigma_garch_wig20, sigma_garch_spx) * np.asarray(vix, dtype=float)


def implied_vol(k, vix_T, params: SkewParams, sigma_garch_wig20, sigma_garch_spx,
                shift_pp: float = 0.0, skew=None) -> np.ndarray:
    """σ_imp,WIG20(k, T); ``vix_T`` – σ_VIX(T) z powierzchni S&P 500 (dziesiętnie)."""
    ratio = vol_ratio(sigma_garch_wig20, sigma_garch_spx)
    sigma = ratio * params.atm(vix_T) + params.slope(vix_T, skew) * np.asarray(k, dtype=float)
    return np.maximum(sigma + shift_pp / 100.0, params.min_sigma)


def sensitivity(k, vix_T, params: SkewParams, sigma_garch_wig20, sigma_garch_spx,
                shift_pp: float = DEFAULT_SHIFT_PP, skew=None) -> dict[str, np.ndarray]:
    """Warianty −δ, 0, +δ zmienności implikowanej WIG20 do analizy wrażliwości."""
    return {f"{s:+.0f}pp" if s else "base": implied_vol(k, vix_T, params, sigma_garch_wig20,
                                                         sigma_garch_spx, s, skew)
            for s in (-shift_pp, 0.0, shift_pp)}
