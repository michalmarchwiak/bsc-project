
import numpy as np
import pandas as pd

from scipy.stats import norm, t


def simulate_portfolio(
    etfs,
    weights,
    model,
    confidence,
    mu,
    sigma,
    correlation,
    residuals_train,
    nu=None,
    n_simulations=10000,
    seed=42
):

    # 1. Sprawdzenie danych wejściowych

    etfs = list(etfs)

    weights = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    R = np.asarray(correlation, dtype=float)

    n = len(etfs)

    if n == 0 or len(set(etfs)) != n:
        raise ValueError("Lista ETF-ów musi być niepusta i bez duplikatów.")

    if (
        len(weights) != n
        or len(mu) != n
        or len(sigma) != n
        or R.shape != (n, n)
    ):
        raise ValueError("Niezgodne wymiary danych wejściowych.")

    if not (
        np.isfinite(weights).all()
        and np.isfinite(mu).all()
        and np.isfinite(sigma).all()
        and np.isfinite(R).all()
    ):
        raise ValueError("Dane zawierają NaN lub infinity.")

    if (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Wagi muszą być nieujemne i sumować się do 1.")

    if (sigma <= 0).any():
        raise ValueError("Odchylenia standardowe muszą być dodatnie.")

    if not 0 < confidence < 1:
        raise ValueError("Nieprawidłowy poziom ufności.")

    if not isinstance(n_simulations, int) or n_simulations < 100:
        raise ValueError("Liczba symulacji musi wynosić co najmniej 100.")

    if model not in [
        "Static_Gaussian",
        "Static_Student",
        "Dynamic_Gaussian"
    ]:
        raise ValueError("Nieznany model kopuły.")

    if not np.allclose(R, R.T, atol=1e-8):
        raise ValueError("Macierz korelacji nie jest symetryczna.")

    if not np.allclose(np.diag(R), 1, atol=1e-8):
        raise ValueError("Przekątna macierzy korelacji musi być równa 1.")

    if np.linalg.eigvalsh(R).min() <= 0:
        raise ValueError("Macierz korelacji nie jest dodatnio określona.")

    if model == "Static_Student":
        if nu is None or not np.isfinite(nu) or nu <= 2:
            raise ValueError("Dla kopuły Studenta wymagane jest nu > 2.")

    if not isinstance(residuals_train, pd.DataFrame):
        raise ValueError("Reszty TRAIN muszą być tabelą pandas.")

    if not set(etfs).issubset(residuals_train.columns):
        raise ValueError("Brakuje kolumn ETF w resztach TRAIN.")

    residuals = residuals_train[etfs].to_numpy(dtype=float)

    if len(residuals) < 2 or not np.isfinite(residuals).all():
        raise ValueError("Nieprawidłowe reszty TRAIN.")

    # 2. Generowanie losowych scenariuszy

    rng = np.random.default_rng(seed)

    independent = rng.standard_normal(
        size=(n_simulations, n)
    )

    L = np.linalg.cholesky(R)

    correlated = independent @ L.T

    # 3. Symulacja wybranej kopuły

    if model == "Static_Student":

        chi_square = rng.chisquare(
            df=nu,
            size=n_simulations
        )

        student_scores = (
            correlated / np.sqrt(chi_square[:, None] / nu)
        )

        simulated_u = t.cdf(
            student_scores,
            df=nu
        )

    else:

        simulated_u = norm.cdf(correlated)

    # 4. Transformacja do empirycznych reszt TRAIN

    sorted_residuals = np.sort(
        residuals,
        axis=0
    )

    quantile_grid = np.linspace(
        0, 1, len(residuals)
    )

    shocks = np.zeros_like(simulated_u)

    for i in range(n):

        shocks[:, i] = np.interp(
            simulated_u[:, i],
            quantile_grid,
            sorted_residuals[:, i]
        )

    # 5. Symulowane logarytmiczne stopy zwrotu
    # mu i sigma są wyrażone w procentach

    simulated_log = (
        mu + shocks * sigma
    )

    # 6. Konwersja na proste stopy zwrotu

    simulated_simple = np.expm1(
        simulated_log / 100
    )

    # 7. Symulowane stopy zwrotu portfela

    portfolio_returns = (
        simulated_simple @ weights
    )

    # 8. Obliczanie VaR i ES

    losses = -portfolio_returns

    var = np.quantile(
        losses,
        confidence
    )

    es = losses[losses >= var].mean()

    # 9. Wynik dla aplikacji

    return {
        "model": model,
        "confidence": confidence,
        "etfs": etfs,
        "weights": weights.tolist(),
        "VaR": float(var),
        "ES": float(es),
        "VaR_Percent": float(var * 100),
        "ES_Percent": float(es * 100),
        "Mean_Return_Percent": float(
            portfolio_returns.mean() * 100
        ),
        "Volatility_Percent": float(
            portfolio_returns.std(ddof=1) * 100
        ),
        "Simulated_Returns": portfolio_returns
    }
