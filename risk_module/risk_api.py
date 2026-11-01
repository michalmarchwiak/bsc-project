
import json
import numpy as np

from risk_pipeline import calculate_risk


def analyze_portfolio(data):

    # 1. Sprawdzenie podstawowych danych

    required = [
        "date",
        "etfs",
        "weights",
        "model",
        "confidence"
    ]

    for field in required:
        if field not in data:
            raise ValueError(
                "Brakuje pola: " + field
            )

    # 2. Uruchomienie silnika ryzyka

    result = calculate_risk(
        date=data["date"],
        etfs=data["etfs"],
        weights=data["weights"],
        model=data["model"],
        confidence=data["confidence"],
        n_simulations=data.get("n_simulations", 10000),
        seed=data.get("seed", 42)
    )

    # 3. Przygotowanie odpowiedzi JSON

    response = {
        "date": result["date"],
        "model": result["model"],
        "confidence": result["confidence"],
        "etfs": result["etfs"],
        "weights": result["weights"],
        "n_simulations": len(
            result["Simulated_Returns"]
        ),
        "risk": {
            "var_percent": result["VaR_Percent"],
            "es_percent": result["ES_Percent"],
            "mean_return_percent": result["Mean_Return_Percent"],
            "volatility_percent": result["Volatility_Percent"]
        },
        "garch": {
            "mean": result["GARCH_Mean"],
            "volatility": result["GARCH_Volatility"]
        },
        "correlation_matrix": result["Correlation_Matrix"]
    }

    # 4. Opcjonalne dołączenie scenariuszy

    if data.get("include_simulations", False):

        response["simulated_returns"] = (
            result["Simulated_Returns"].tolist()
        )

    # 5. Sprawdzenie serializacji JSON

    json.dumps(
        response,
        allow_nan=False
    )

    return response
