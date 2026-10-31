"""Ocena modeli zależności: AIC/BIC, test Vuonga i log-score kopuły poza próbą.

Test Vuonga (1989) z poprawką Schwarza dla modeli niezagnieżdżonych f₁, f₂ (k₁, k₂ parametrów),
na tych samych pseudo-obserwacjach okna estymacji (n obserwacji):
    mᵢ = log c₁(uᵢ) − log c₂(uᵢ),
    Z = [Σᵢ mᵢ − (k₁ − k₂)·ln(n)/2] / (√n · s_m),   s_m – odchylenie standardowe mᵢ,
pod H0 (modele równoważne) Z ~ N(0, 1). Z > z_{1−α/2} – lepszy model 1, Z < −z_{1−α/2} – model 2.

Log-score poza próbą: u_t = F̂_i,t(r_i,t) z filtru GARCH naprzód parametrami z okna
(``MarginFit.pit_forward``) dla sesji między końcem okna a końcem następnego okna; dni zamknięcia
całego rynku są usuwane, pojedyncze braki → u = F_skewt(0) (odpowiednik ẑ = 0 w panelu reszt).
    LS = Σ_t log ĉ(u_t)   (wyższy – lepszy; kopuła empiryczna bez gęstości jest pomijana).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pf.models.dependence import DependenceModel
from pf.models.margins import MarginFit, skewt_cdf


def default_market(ticker: str) -> str:
    return "PL" if ticker.endswith(".WA") else "US"


def information_criteria(model: DependenceModel) -> dict:
    return {"model": model.name, "loglik": model.loglik, "n_params": model.n_params,
            "aic": model.aic(), "bic": model.bic(), "n_obs": model.n_obs}


def vuong_test(ll1: np.ndarray, ll2: np.ndarray, k1: int, k2: int, level: float = 0.05) -> dict:
    """Test Vuonga z poprawką Schwarza; ``winner`` = 1 (model 1), 2 (model 2) lub 0 (nierozstrzygnięty)."""
    m = np.asarray(ll1, dtype=float) - np.asarray(ll2, dtype=float)
    n = len(m)
    sd = m.std(ddof=1)
    corr = (k1 - k2) * np.log(n) / 2.0
    if not np.isfinite(sd) or sd <= 0:
        stat = 0.0 if abs(m.sum() - corr) < 1e-12 else np.sign(m.sum() - corr) * np.inf
    else:
        stat = (m.sum() - corr) / (np.sqrt(n) * sd)
    p = float(2.0 * stats.norm.sf(abs(stat)))
    crit = stats.norm.ppf(1.0 - level / 2.0)
    winner = 1 if stat > crit else (2 if stat < -crit else 0)
    return {"stat": float(stat), "pvalue": p, "winner": winner, "mean_diff": float(m.mean()),
            "correction": float(corr), "n": n}


def compare_with_reference(models: dict[str, DependenceModel], U: np.ndarray, reference: str = "rvine",
                           level: float = 0.05) -> pd.DataFrame:
    """AIC/BIC wszystkich modeli i test Vuonga ``reference`` vs każdy pozostały model z gęstością."""
    rows = []
    ref = models.get(reference)
    ll_ref = ref.logpdf(U) if ref is not None and ref.has_density else None
    for name, m in models.items():
        row = information_criteria(m)
        row["model"] = name
        if ll_ref is not None and name != reference and m.has_density:
            v = vuong_test(ll_ref, m.logpdf(U), ref.n_params, m.n_params, level)
            row.update({"vuong_stat": v["stat"], "vuong_pvalue": v["pvalue"],
                        "vuong_winner": {1: reference, 2: name, 0: "brak"}[v["winner"]]})
        rows.append(row)
    return pd.DataFrame(rows)


def pit_panel(margins: dict[str, MarginFit], r_oos: pd.DataFrame, market_of=default_market) -> pd.DataFrame:
    """Panel u_t (sesje × aktywa) z filtru GARCH naprzód; zasady panelu jak w ``residual_panel``."""
    assets = [a for a in r_oos.columns if a in margins]
    U = pd.DataFrame({a: margins[a].pit_forward(r_oos[a].to_numpy()) for a in assets}, index=r_oos.index)
    markets = pd.Series({a: market_of(a) for a in assets})
    open_all = np.ones(len(U), dtype=bool)
    for mk in markets.unique():
        open_all &= U[markets.index[markets == mk]].notna().any(axis=1).to_numpy()
    U = U[open_all]
    fill = {a: float(skewt_cdf(np.array([0.0]), margins[a].eta, margins[a].lam)[0]) for a in assets}
    return U.fillna(fill)


def oos_log_score(model: DependenceModel, U_oos: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Dzienne log ĉ(u_t) poza próbą (kolumny w kolejności ``model.names``)."""
    if not model.has_density:
        raise ValueError(f"Model {model.name} nie ma gęstości")
    X = U_oos[model.names].to_numpy() if isinstance(U_oos, pd.DataFrame) else np.asarray(U_oos)
    return model.logpdf(X)
