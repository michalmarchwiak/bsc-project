# Analiza ryzyka funduszu inwestycyjnego z uwzględnieniem zależności aktywów

Projekt inżynierski zespołu czteroosobowego. Celem jest analiza wpływu zależności między funduszami ETF na ryzyko portfela oraz przygotowanie aplikacji umożliwiającej porównywanie strategii i modeli ryzyka.

> **Stan prac: 1 listopada 2026 r.** Część danych, modelowania i badań VaR została zrealizowana i przetestowana. Udostępniany moduł jest **prototypem badawczym**, a nie narzędziem do bieżących rekomendacji inwestycyjnych. Podstawowy ekran Streamlit został lokalnie uruchomiony; integracja aplikacji zespołowej pozostaje do wykonania.

## 1. Co jest zrobione

**Dane i przetwarzanie:**
- 10 ETF-ów: `SPY`, `QQQ`, `EFA`, `EEM`, `IEF`, `TLT`, `LQD`, `HYG`, `GLD`, `VNQ`.
- Sprowadzenie danych do wspólnego kalendarza, kontrola braków i duplikatów, stopy zwrotu proste i logarytmiczne, warianty USD i PLN oraz kursy NBP USD/PLN.
- Wspólna historia dziennych stóp zwrotu: 4 899 sesji, 2007-04-12–2026-09-30.
- Chronologiczny podział: TRAIN 3 204 sesje (2007-04-12–2019-12-31), VALIDATION 756 (2020-01-02–2022-12-30), TEST 939 (2023-01-03–2026-09-30).
- Walidacja danych oraz testy podstawowych przekształceń.

**Modelowanie i badania:**
- Jednowymiarowe modele Constant-GARCH(1,1) z innowacjami t-Studenta, estymowane na TRAIN.
- Pseudoobserwacje oraz kopuły: statyczna Gaussa, statyczna t-Studenta i Gaussa z macierzą korelacji aktualizowaną z ostatnich 252 sesji.
- Symulacje Monte Carlo oraz jednodniowe VaR i ES dla portfeli Equal Weight, Equity Heavy i Defensive.
- Backtesting VaR (Kupiec UC, Christoffersen IND i CC), analiza przekroczeń, porównanie Quantile Loss na zbiorze TEST.
- Weryfikacja zgodności liczbowej modułu aplikacyjnego z backtestingiem dla przykładowej sesji.

**Moduł aplikacyjny:**
- `risk_engine.py` — symulacja scenariuszy i VaR/ES.
- `risk_pipeline.py` — prognozy GARCH i macierze korelacji na wskazaną historyczną datę TEST.
- `risk_api.py` — funkcja `analyze_portfolio(request)` zwracająca słownik gotowy do serializacji JSON.
- `example_integration.py` i `example_response.json` — przykład integracji.
- `README_RISK_API.md` — szczegółowa instrukcja użycia modułu.
- Pierwszy interfejs `app.py` w Streamlit został lokalnie uruchomiony, lecz nie zastępuje docelowej aplikacji zespołowej.

## 2. Najważniejsze wyniki empiryczne

Na **TEST (939 sesji, 2023–2026)** uzyskano następujące liczby przekroczeń VaR (w nawiasie: liczba przypadków, gdy test CC nie odrzucił hipotezy przy poziomie istotności 5%):

| Strategia | Model | Przekroczenia VaR 95% | Przekroczenia VaR 99% |
|---|---|---:|---:|
| Defensive | Static Gaussian | 81 | 33 |
| Defensive | Static Student | 82 | 29 |
| Defensive | Dynamic Gaussian | **51** | **14** |
| Equal Weight | Static Gaussian | 68 | 19 |
| Equal Weight | Static Student | 71 | 18 |
| Equal Weight | Dynamic Gaussian | **39** | **11** |
| Equity Heavy | Static Gaussian | 44 | 9 |
| Equity Heavy | Static Student | 46 | 7 |
| Equity Heavy | Dynamic Gaussian | 39 | 6 |

Oczekiwana liczba przekroczeń wynosi około 47 (VaR 95%) lub 9 (VaR 99%). Model dynamiczny uzyskał **5/6 nieodrzuconych hipotez CC**, a każdy z modeli statycznych **2/6**. Dla Defensive model dynamiczny uzyskał również najniższy Quantile Loss przy obu poziomach VaR. Dla Equity Heavy wyniki modeli statycznych były konkurencyjne. Nieodrzucenie hipotezy nie dowodzi poprawności modelu ani istotnej przewagi nad innym modelem.

**Ograniczenia:** brak formalnego backtestingu ES; rozkłady brzegowe nie są idealnie skalibrowane; dynamiczny model używa 252-sesyjnego okna; wielowymiarowa kopuła t-Studenta korzysta z macierzy R z modelu Gaussa (bez pełnej wspólnej optymalizacji); wyniki są w USD i dotyczą jednego dnia.

Szczegóły, zadania do przejęcia i kryteria ukończenia: [HANDOFF_DLA_ZESPOLU.md](HANDOFF_DLA_ZESPOLU.md).

## 3. Szybki start

Wymagany Python oraz pakiety `numpy`, `pandas`, `scipy`, `arch`.

```bash
cd risk_module
python3 -m pip install -r requirements.txt
python3 example_integration.py
```

Wywołanie w kodzie Pythona:

```python
from risk_api import analyze_portfolio

request = {
    "date": "2023-01-03",
    "etfs": ["SPY", "QQQ", "IEF", "TLT"],
    "weights": [0.25, 0.25, 0.25, 0.25],
    "model": "Dynamic_Gaussian",
    "confidence": 0.99,
    "n_simulations": 10000,
    "seed": 42,
}

result = analyze_portfolio(request)
print(result["risk"])
```

**Uwaga:** funkcja obsługuje tylko daty notowań z TEST (2023-01-03–2026-09-30). Parametry GARCH i kopuł są wyznaczone na TRAIN. Stany GARCH i korelacje kroczące korzystają wyłącznie z historii dostępnej przed badaną sesją. W obecnym kodzie ścieżki do danych są względne — polecenia należy uruchamiać z katalogu `risk_module/`.

### Streamlit — prototyp (jeśli plik `app.py` jest w repozytorium)

```bash
cd risk_module
python3 -m pip install streamlit
python3 -m streamlit run app.py
```

`app.py` nie został automatycznie dołączony do wcześniej przygotowanej paczki `risk_module.zip` — jeśli chcemy udostępnić prototyp, trzeba go skopiować do `risk_module/` i dopiero wtedy spakować lub zatwierdzić w repozytorium.

## 4. Struktura repozytorium

```text
repo/
├── README.md
├── HANDOFF_DLA_ZESPOLU.md
├── .gitignore
└── risk_module/
    ├── risk_engine.py
    ├── risk_pipeline.py
    ├── risk_api.py
    ├── example_integration.py
    ├── example_response.json
    ├── README_RISK_API.md
    ├── requirements.txt
    └── data/processed/
        ├── splits/returns_log_usd_{train,validation,test}.csv
        ├── standardized_residuals_train.csv
        ├── gaussian_copula_correlation_matrix.csv
        ├── multivariate_t_copula_parameters.csv
        └── copula_pseudo_observations_{train,validation,test}.csv
```

Uwaga: zapis z nawiasami klamrowymi powyżej skraca listę faktycznych plików; w katalogu są trzy osobne pliki w każdym takim przypadku.

## 5. Co dalej

1. Umieścić **pliki źródłowe** `risk_module/` w GitHubie, a nie wyłącznie archiwum ZIP. ZIP może być dodatkowym artefaktem do pobrania.
2. Zintegrować `analyze_portfolio()` z docelową aplikacją Streamlit, w szczególności obsługę wyboru ETF, wag, sesji, modelu i poziomu ufności.
3. Rozszerzyć walidację wejścia, testy integracyjne i obsługę błędów w UI; ustalić sposób współdzielenia wyników i konfiguracji.
4. Jeśli celem jest prognozowanie na **bieżący dzień**, dobudować osobny pipeline aktualizacji danych, walidację i harmonogram estymacji (obecne API tego nie robi).
5. Dopracować wykresy i raport końcowy; przeprowadzić przegląd metodyki i ograniczeń modeli. Zapisane wyniki TEST należy traktować jako końcową ocenę, nie jako kolejny zbiór do dostrajania parametrów.

## 6. Zasady współpracy

- Pracujemy na oddzielnych gałęziach i łączymy zmiany przez Pull Request.
- Do zmian algorytmicznych dołączamy testy oraz opis, czy zmieniają wyniki VaR/ES.
- Nie dodajemy lokalnych środowisk `.venv`, cache, plików systemowych ani sekretów.
- Jeśli historyczne dane ETF mają ograniczenia licencyjne, przed publicznym udostępnieniem należy zweryfikować warunki ich redystrybucji; ewentualnie użyć prywatnego repozytorium lub skryptu pobierania danych.
