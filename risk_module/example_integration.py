
import json

from risk_api import analyze_portfolio


def main():

    # Dane podawane przez użytkownika aplikacji

    request = {
        "date": "2023-01-03",
        "etfs": ["SPY", "QQQ", "IEF", "TLT"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "model": "Dynamic_Gaussian",
        "confidence": 0.99,
        "n_simulations": 10000,
        "seed": 42
    }

    # Wywołanie naszego modułu

    result = analyze_portfolio(request)

    # Wyniki do wyświetlenia w aplikacji

    risk = result["risk"]

    print("\nANALIZA RYZYKA PORTFELA")

    print("Data:", result["date"])
    print("ETF-y:", ", ".join(result["etfs"]))
    print("Model:", result["model"])

    print("\nWYNIKI")

    print("VaR 99%:", round(risk["var_percent"], 4), "%")
    print("ES 99%:", round(risk["es_percent"], 4), "%")

    print(
        "Średni zwrot:",
        round(risk["mean_return_percent"], 4),
        "%"
    )

    print(
        "Zmienność:",
        round(risk["volatility_percent"], 4),
        "%"
    )

    # Zapis odpowiedzi do JSON

    with open("example_response.json", "w") as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    print("\nZapisano example_response.json")


if __name__ == "__main__":
    main()
