
import numpy as np
import pandas as pd
import streamlit as st

from risk_api import analyze_portfolio


# 1. Konfiguracja aplikacji

st.set_page_config(
    page_title="Risk Portfolio Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("Risk Portfolio Analyzer")

st.write(
    "Analiza ryzyka portfela inwestycyjnego "
    "z uwzględnieniem zależności między aktywami."
)


# 2. Dostępne fundusze ETF

all_etfs = [
    "SPY", "QQQ", "EFA", "EEM", "IEF",
    "TLT", "LQD", "HYG", "GLD", "VNQ"
]


# 3. Dostępne daty historyczne TEST

@st.cache_data
def load_dates():

    data = pd.read_csv(
        "data/processed/splits/returns_log_usd_test.csv",
        usecols=["Date"]
    )

    return data["Date"].tolist()


dates = load_dates()


# 4. Panel boczny

st.sidebar.header("Ustawienia portfela")

selected_etfs = st.sidebar.multiselect(
    "Wybierz fundusze ETF",
    all_etfs,
    default=["SPY", "QQQ", "IEF", "TLT"]
)

st.sidebar.subheader("Wagi portfela")

weights = []

if len(selected_etfs) > 0:

    for etf in selected_etfs:

        weight = st.sidebar.number_input(
            f"Waga {etf}",
            min_value=0.0,
            max_value=1.0,
            value=round(1.0 / len(selected_etfs), 4),
            step=0.05,
            format="%.4f",
            key="weight_" + etf
        )

        weights.append(weight)

weight_sum = sum(weights)

st.sidebar.write(
    "Suma wag:",
    round(weight_sum * 100, 2),
    "%"
)


# 5. Model zależności

model = st.sidebar.selectbox(
    "Model kopuły",
    [
        "Static_Gaussian",
        "Static_Student",
        "Dynamic_Gaussian"
    ],
    index=2
)


# 6. Poziom ufności

confidence = st.sidebar.selectbox(
    "Poziom ufności",
    [0.95, 0.99],
    index=1,
    format_func=lambda x: f"{int(x * 100)}%"
)


# 7. Data prognozy

date = st.sidebar.selectbox(
    "Data analizy",
    dates,
    index=0
)


# 8. Liczba symulacji

n_simulations = st.sidebar.selectbox(
    "Liczba symulacji Monte Carlo",
    [1000, 5000, 10000, 20000],
    index=2
)


# 9. Uruchomienie analizy

analyze = st.sidebar.button(
    "Oblicz ryzyko",
    type="primary"
)

if analyze:

    if len(selected_etfs) == 0:

        st.error(
            "Wybierz przynajmniej jeden fundusz ETF."
        )

    elif not np.isclose(weight_sum, 1.0, atol=1e-6):

        st.error(
            "Suma wag portfela musi wynosić 100%."
        )

    else:

        request = {
            "date": date,
            "etfs": selected_etfs,
            "weights": weights,
            "model": model,
            "confidence": confidence,
            "n_simulations": n_simulations,
            "seed": 42,
            "include_simulations": True
        }

        try:

            with st.spinner(
                "Obliczanie ryzyka portfela..."
            ):

                result = analyze_portfolio(request)

            st.session_state["result"] = result

        except (ValueError, TypeError) as error:

            st.error(str(error))
            st.session_state.pop("result", None)

        except Exception:

            st.error(
                "Wystąpił błąd podczas analizy. "
                "Sprawdź pliki danych i konfigurację."
            )
            st.session_state.pop("result", None)


# 10. Prezentacja wyników

if "result" in st.session_state:

    result = st.session_state["result"]

    risk = result["risk"]

    st.subheader("Wyniki analizy portfela")

    st.caption(
        f"Data: {result['date']} | "
        f"Model: {result['model']} | "
        f"Poziom ufności: {result['confidence']:.0%}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Value at Risk",
        f"{risk['var_percent']:.4f}%"
    )

    col2.metric(
        "Expected Shortfall",
        f"{risk['es_percent']:.4f}%"
    )

    col3.metric(
        "Średni zwrot",
        f"{risk['mean_return_percent']:.4f}%"
    )

    col4.metric(
        "Zmienność",
        f"{risk['volatility_percent']:.4f}%"
    )

    # 11. Rozkład stóp zwrotu

    st.subheader("Rozkład symulowanych stóp zwrotu")

    returns = np.array(
        result["simulated_returns"]
    ) * 100

    counts, edges = np.histogram(
        returns,
        bins=50
    )

    histogram = pd.DataFrame({
        "Stopa zwrotu (%)": (
            edges[:-1] + edges[1:]
        ) / 2,
        "Liczba scenariuszy": counts
    })

    st.bar_chart(
        histogram,
        x="Stopa zwrotu (%)",
        y="Liczba scenariuszy"
    )

    st.caption(
        "Rozkład jednodniowych stóp zwrotu "
        "wygenerowanych metodą Monte Carlo."
    )

    # 12. Skład portfela

    st.subheader("Skład portfela")

    portfolio = pd.DataFrame({
        "ETF": result["etfs"],
        "Waga (%)": [
            weight * 100
            for weight in result["weights"]
        ]
    })

    st.dataframe(
        portfolio,
        hide_index=True
    )

    # 13. Macierz korelacji

    st.subheader("Macierz korelacji")

    correlation = pd.DataFrame(
        result["correlation_matrix"],
        index=result["etfs"],
        columns=result["etfs"]
    )

    st.dataframe(
        correlation.style.format("{:.4f}")
    )

    # 14. Parametry GARCH

    st.subheader("Prognozy GARCH")

    garch = pd.DataFrame({
        "ETF": result["etfs"],
        "Średnia (%)": result["garch"]["mean"],
        "Zmienność (%)": result["garch"]["volatility"]
    })

    st.dataframe(
        garch.style.format({
            "Średnia (%)": "{:.4f}",
            "Zmienność (%)": "{:.4f}"
        }),
        hide_index=True
    )

else:

    st.info(
        "Wybierz ETF-y, ustaw wagi i kliknij "
        "'Oblicz ryzyko', aby rozpocząć analizę."
    )
