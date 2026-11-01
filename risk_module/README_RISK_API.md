
# Risk Analysis Module

## 1. Opis

Moduł służy do analizy jednodniowego ryzyka portfela
inwestycyjnego złożonego z wybranych funduszy ETF.

Wykorzystuje modele GARCH(1,1), kopuły oraz symulacje
Monte Carlo do wyznaczania VaR i Expected Shortfall.

## 2. Dostępne modele

- Static_Gaussian — statyczna kopuła Gaussa
- Static_Student — statyczna kopuła t-Studenta
- Dynamic_Gaussian — kopuła Gaussa z korelacją
  kroczącą z poprzednich 252 sesji

## 3. Dostępne ETF-y

SPY, QQQ, EFA, EEM, IEF, TLT, LQD, HYG, GLD, VNQ

Wszystkie prognozy dotyczą stóp zwrotu w USD.

## 4. Struktura modułu

risk_engine.py
- Generowanie scenariuszy Monte Carlo
- Symulowanie zależnych stóp zwrotu aktywów
- Obliczanie VaR i ES

risk_pipeline.py
- Wczytywanie danych
- Prognozy GARCH
- Wyznaczanie macierzy korelacji
- Uruchamianie silnika symulacyjnego

risk_api.py
- Obsługa danych wejściowych
- Wywoływanie analizy
- Przygotowanie odpowiedzi JSON

## 5. Wymagane biblioteki

numpy
pandas
scipy
arch

Instalacja:

python3 -m pip install numpy pandas scipy arch

## 6. Przykład użycia

```python
from risk_api import analyze_portfolio

request = {
    "date": "2023-01-03",
    "etfs": ["SPY", "QQQ", "IEF", "TLT"],
    "weights": [0.25, 0.25, 0.25, 0.25],
    "model": "Dynamic_Gaussian",
    "confidence": 0.99,
    "n_simulations": 10000,
    "seed": 42
}

result = analyze_portfolio(request)

print(result["risk"])
```

## 7. Parametry wejściowe

date:
Data jednodniowej prognozy w formacie YYYY-MM-DD.
Aktualna wersja obsługuje wyłącznie sesje ze zbioru
TEST od 2023-01-03 do 2026-09-30.

etfs:
Lista wybranych ETF-ów bez duplikatów.

weights:
Lista nieujemnych wag portfela.
Liczba wag musi odpowiadać liczbie ETF-ów.
Suma wag musi wynosić 1.

model:
Jeden z trzech dostępnych modeli.

confidence:
Poziom ufności, np. 0.95 lub 0.99.

n_simulations:
Liczba scenariuszy Monte Carlo, minimum 100.
Domyślnie 10000.

seed:
Ziarno generatora liczb losowych.
Domyślnie 42.

include_simulations:
Opcjonalna wartość logiczna.
Domyślnie False.
Jeśli True, odpowiedź zawiera wszystkie
symulowane stopy zwrotu portfela.

## 8. Dane wyjściowe

Funkcja analyze_portfolio() zwraca słownik
zawierający:

date
model
confidence
etfs
weights
n_simulations

risk:
- var_percent
- es_percent
- mean_return_percent
- volatility_percent

garch:
- mean
- volatility

correlation_matrix

Opcjonalnie:
- simulated_returns

## 9. Interpretacja VaR i ES

VaR 99% jest prognozowanym progiem jednodniowej
straty, którego przekroczenie ma według modelu
prawdopodobieństwo 1%.

ES 99% jest średnią stratą w najgorszym 1%
symulowanych scenariuszy.

Wartości VaR i ES są wyrażone w procentach
wartości portfela.

## 10. Metodologia

Modele GARCH i statyczne kopuły oszacowano
na danych TRAIN.

Parametry GARCH pozostają stałe.
Warunkowa wariancja jest aktualizowana na podstawie
historycznych stóp zwrotu dostępnych przed
prognozowaną sesją.

Dynamiczna macierz korelacji korzysta z 252
poprzednich sesji.

Rozkłady brzegowe są oparte na empirycznych
rozkładach standaryzowanych reszt z TRAIN.

Kopuła t-Studenta wykorzystuje stałą macierz
zależności z kopuły Gaussa i wspólny parametr nu.

## 11. Ograniczenia

- Moduł nie pobiera bieżących danych rynkowych.
- Obsługuje wyłącznie historyczne daty TEST.
- Nie przeprowadza ponownej estymacji parametrów
  GARCH ani kopuł w czasie prognozowania.
- Wszystkie wyniki dotyczą jednodniowego horyzontu.
- Wagi portfela są nieujemne i sumują się do 1.
- Prognozy są oparte na danych w USD.
- ES nie został poddany osobnemu formalnemu
  backtestingowi.
- Wyniki są prognozami modelowymi, nie gwarancjami.

## 12. Testowanie

python3 test_risk_engine.py
python3 test_risk_pipeline.py
python3 test_pipeline_consistency.py
python3 test_risk_api.py
python3 test_risk_api_validation.py

## 13. Integracja

Główny punkt wejścia:

from risk_api import analyze_portfolio

Wywołanie:

response = analyze_portfolio(request)

Funkcja zwraca słownik Pythona, który można
serializować do JSON.

Obecnie moduł nie udostępnia serwera HTTP.
Może zostać podłączony do aplikacji Python,
np. FastAPI lub Streamlit.

## 14. Wymagane dane

data/processed/splits/
- returns_log_usd_train.csv
- returns_log_usd_validation.csv
- returns_log_usd_test.csv

data/processed/
- standardized_residuals_train.csv
- gaussian_copula_correlation_matrix.csv
- multivariate_t_copula_parameters.csv
- copula_pseudo_observations_train.csv
- copula_pseudo_observations_validation.csv
- copula_pseudo_observations_test.csv

Pliki muszą znajdować się w odpowiednich
katalogach względem głównego folderu projektu.

## 15. Uwagi

Moduł jest prototypem badawczym przygotowanym
w ramach projektu inżynierskiego.

Przed wykorzystaniem w środowisku produkcyjnym
zalecane jest dodanie automatycznego pobierania
danych, buforowania modeli, dodatkowych testów
oraz obsługi błędów po stronie interfejsu.
```
