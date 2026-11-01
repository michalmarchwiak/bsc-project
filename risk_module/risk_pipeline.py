
import numpy as np
import pandas as pd

from arch import arch_model
from scipy.stats import norm

from risk_engine import simulate_portfolio


ALL_ETFS = [
    "SPY", "QQQ", "EFA", "EEM", "IEF",
    "TLT", "LQD", "HYG", "GLD", "VNQ"
]


def calculate_risk(
    date,
    etfs,
    weights,
    model,
    confidence=0.99,
    n_simulations=10000,
    seed=42
):

    # 1. Kontrola wybranych ETF-ów

    etfs = list(etfs)

    if len(etfs) == 0 or len(set(etfs)) != len(etfs):
        raise ValueError("Podaj niepustą listę ETF-ów bez duplikatów.")

    if not set(etfs).issubset(ALL_ETFS):
        raise ValueError("Wybrano ETF spoza dostępnej listy.")

    date = pd.Timestamp(date)

    if pd.isna(date):
        raise ValueError("Nieprawidłowa data.")

    # 2. Wczytanie stóp zwrotu

    folder = "data/processed/splits/"

    train = pd.read_csv(
        folder + "returns_log_usd_train.csv",
        index_col=0,
        parse_dates=True
    )[etfs]

    validation = pd.read_csv(
        folder + "returns_log_usd_validation.csv",
        index_col=0,
        parse_dates=True
    )[etfs]

    test = pd.read_csv(
        folder + "returns_log_usd_test.csv",
        index_col=0,
        parse_dates=True
    )[etfs]

    if date not in test.index:
        raise ValueError(
            "Data musi należeć do zbioru TEST."
        )

    # 3. Reszty GARCH z TRAIN

    residuals = pd.read_csv(
        "data/processed/standardized_residuals_train.csv",
        index_col=0,
        parse_dates=True
    )[etfs]

    assert train.index.equals(residuals.index)

    # 4. Stała macierz zależności

    R_static = pd.read_csv(
        "data/processed/gaussian_copula_correlation_matrix.csv",
        index_col=0
    ).loc[etfs, etfs].to_numpy()

    R = R_static.copy()

    # 5. Dynamiczna macierz korelacji

    if model == "Dynamic_Gaussian":

        pseudo_train = pd.read_csv(
            "data/processed/copula_pseudo_observations_train.csv",
            index_col=0,
            parse_dates=True
        )[etfs]

        pseudo_validation = pd.read_csv(
            "data/processed/copula_pseudo_observations_validation.csv",
            index_col=0,
            parse_dates=True
        )[etfs]

        pseudo_test = pd.read_csv(
            "data/processed/copula_pseudo_observations_test.csv",
            index_col=0,
            parse_dates=True
        )[etfs]

        all_pseudo = pd.concat([
            pseudo_train,
            pseudo_validation,
            pseudo_test
        ])

        assert all_pseudo.index.is_unique
        assert all_pseudo.index.is_monotonic_increasing

        # Pobieramy tylko sesje wcześniejsze niż date

        history = all_pseudo.loc[
            all_pseudo.index < date
        ].tail(252)

        if len(history) != 252:
            raise ValueError(
                "Za mało obserwacji do korelacji dynamicznej."
            )

        values = history.to_numpy()

        if not ((values > 0) & (values < 1)).all():
            raise ValueError("Nieprawidłowe pseudoobserwacje.")

        # Transformacja PIT

        Z = pd.DataFrame(
            norm.ppf(values),
            index=history.index,
            columns=etfs
        )

        R = Z.corr().to_numpy()

    # 6. Parametr kopuły t-Studenta

    parameters = pd.read_csv(
        "data/processed/multivariate_t_copula_parameters.csv"
    )

    nu = float(parameters["Nu"].iloc[0])

    # 7. Parametry i prognozy GARCH

    mu = []
    sigma = []

    for etf in etfs:

        train_returns = train[etf] * 100

        garch = arch_model(
            train_returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="t"
        )

        fit = garch.fit(disp="off")

        if fit.convergence_flag != 0:
            raise ValueError(
                "Brak zbieżności GARCH dla " + etf
            )

        mean = fit.params["mu"]
        omega = fit.params["omega"]
        alpha = fit.params["alpha[1]"]
        beta = fit.params["beta[1]"]

        previous_return = train_returns.iloc[-1]

        previous_variance = (
            fit.conditional_volatility.iloc[-1] ** 2
        )

        # Historia po TRAIN, ale wyłącznie przed date

        historical_returns = pd.concat([
            validation[etf],
            test[etf]
        ])

        historical_returns = historical_returns.loc[
            historical_returns.index < date
        ] * 100

        # Aktualizacja stanu GARCH
        # Nie zmieniamy oszacowanych parametrów

        for actual_return in historical_returns:

            previous_residual = previous_return - mean

            variance = (
                omega
                + alpha * previous_residual ** 2
                + beta * previous_variance
            )

            previous_variance = variance
            previous_return = actual_return

        # Prognoza dla wybranej daty

        previous_residual = previous_return - mean

        forecast_variance = (
            omega
            + alpha * previous_residual ** 2
            + beta * previous_variance
        )

        if not np.isfinite(forecast_variance) or forecast_variance <= 0:
            raise ValueError(
                "Nieprawidłowa wariancja dla " + etf
            )

        mu.append(mean)
        sigma.append(np.sqrt(forecast_variance))

    # 8. Wywołanie silnika Monte Carlo

    result = simulate_portfolio(
        etfs=etfs,
        weights=weights,
        model=model,
        confidence=confidence,
        mu=mu,
        sigma=sigma,
        correlation=R,
        residuals_train=residuals,
        nu=nu,
        n_simulations=n_simulations,
        seed=seed
    )

    # 9. Dodatkowe informacje

    result["date"] = str(date.date())

    result["GARCH_Mean"] = list(
        map(float, mu)
    )

    result["GARCH_Volatility"] = list(
        map(float, sigma)
    )

    result["Correlation_Matrix"] = R.tolist()

    return result
