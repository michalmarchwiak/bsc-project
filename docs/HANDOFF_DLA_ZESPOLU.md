# Przekazanie modułu analitycznego — status prac i dalsze zadania

**Projekt:** Analiza ryzyka funduszu inwestycyjnego z uwzględnieniem zależności aktywów  
**Data przekazania:** 2026-11-01  
**Status:** część danych, badań i prototyp modułu analitycznego przygotowana; integracja z docelową aplikacją zespołową do realizacji.

## A. Co przekazuję

| Zakres | Co wykonano | Gdzie szukać |
|---|---|---|
| Dane | 10 ETF, synchronizacja, przeliczenia USD/PLN, stopy zwrotu, walidacje, podział TRAIN/VALIDATION/TEST | skrypty badawcze w lokalnym projekcie; nie wszystkie są w `risk_module/` |
| Modele | GARCH(1,1)-t; kopuły: statyczna Gaussa, statyczna t-Studenta, Gaussa z korelacją kroczącą (252 sesje) | `risk_engine.py`, `risk_pipeline.py`, pliki parametrów w `risk_module/data/processed/` |
| Miary ryzyka | jednodniowe VaR i ES z 10 000 symulacji Monte Carlo | `risk_engine.py` |
| Testy jakości | backtesting VaR 95%/99% (Kupiec, Christoffersen), Quantile Loss i porównanie na TEST | wyniki eksperymentów zapisane w lokalnym `data/processed/`; **nie wszystkie zostały spakowane do `risk_module.zip`** |
| API | `analyze_portfolio(request)` → słownik wyników możliwy do zapisania jako JSON | `risk_api.py` |
| Integracja | przykładowe wywołanie i odpowiedź JSON, README | `example_integration.py`, `example_response.json`, `README_RISK_API.md` |
| UI | lokalnie uruchomiony pierwszy ekran Streamlit (wybór ETF i wag, modelu, daty, VaR/ES, histogram, korelacje, GARCH) | `app.py` w lokalnym projekcie; **nie znalazł się w zweryfikowanym archiwum z Kroku 133** |

### Granice gotowości

- **Gotowe:** działający prototyp funkcji Python i niezależny pakiet `risk_module/` dla historycznych dat TEST, zweryfikowany po spakowaniu i rozpakowaniu.
- **Jeszcze niegotowe:** pełna docelowa aplikacja Streamlit, automatyczne aktualne dane rynkowe, optymalizacja wydajności, wdrożenie i obsługa wielu użytkowników.
- **Nie należy twierdzić:** że ES przeszedł formalny backtesting, że testy dowiodły istotnej przewagi statystycznej modelu dynamicznego albo że aplikacja prognozuje bieżące notowania.

## B. API — jak je wywołać

```python
from risk_api import analyze_portfolio

request = {
    "date": "2023-01-03",  # sesja w historycznym TEST
    "etfs": ["SPY", "QQQ", "IEF", "TLT"],
    "weights": [0.25, 0.25, 0.25, 0.25],
    "model": "Dynamic_Gaussian",  # albo Static_Gaussian, Static_Student
    "confidence": 0.99,
    "n_simulations": 10000,
    "seed": 42,
    "include_simulations": False,
}

answer = analyze_portfolio(request)
print(answer["risk"])
```

Odpowiedź zawiera m.in. `risk.var_percent`, `risk.es_percent`, `risk.mean_return_percent`, `risk.volatility_percent`, parametry `garch`, `correlation_matrix`, a opcjonalnie `simulated_returns`. Wagi nieujemne, suma 1; dostępne ETF-y są ograniczone do 10 ustalonych tickerów.

Uruchomienie:

```bash
cd risk_module
python3 -m pip install -r requirements.txt
python3 example_integration.py
```

## C. Wyniki, które warto wykorzystać w raporcie

**Podział danych:** 4 899 zwrotów; TRAIN 3 204; VALIDATION 756; TEST 939.

**TEST 2023–2026, liczba przekroczeń VaR** (odpowiednio 95% / 99%):

| Strategia | Gauss statyczny | Student statyczny | Gauss dynamiczny |
|---|---:|---:|---:|
| Defensive | 81 / 33 | 82 / 29 | **51 / 14** |
| Equal Weight | 68 / 19 | 71 / 18 | **39 / 11** |
| Equity Heavy | 44 / 9 | 46 / 7 | 39 / 6 |

**Test Christoffersena CC (brak odrzucenia H0 przy 5%):** dynamiczna Gaussa 5 z 6; statyczna Gaussa 2 z 6; statyczna Studenta 2 z 6. Dla Equity Heavy kopuły statyczne są konkurencyjne.

**Quantile Loss:** dynamiczna Gaussa najniższa dla Defensive przy VaR 95% i 99%; w pozostałych portfelach przewaga zależy od poziomu ufności.

**Stabilność zależności:** korelacja normalnych wyników PIT `SPY–TLT` zmieniła się z `-0.4141` (TRAIN) do `+0.0786` (2022); `SPY–IEF` z `-0.4276` do `+0.1710`. To uzasadnia badanie korelacji kroczącej, ale nie stanowi formalnego testu stabilności.

## D. Co powinien zrobić zespół — proponowany backlog

| Priorytet | Zadanie | Warunek ukończenia |
|---|---|---|
| P0 | Włączyć `risk_module/` jako katalog kodu do wspólnego repo | `example_integration.py` działa po sklonowaniu repo i instalacji zależności |
| P0 | Przenieść istniejący `app.py` do repo i połączyć z docelowym widokiem Streamlit | z UI da się podać ETF, wagi, datę TEST, model, poziom VaR i otrzymać wynik |
| P0 | Przejąć i opisać odpowiedzialność za pliki wejściowe i format danych | dokumentacja wskazuje, skąd pochodzą pliki i kto je aktualizuje |
| P1 | Testy integracyjne, walidacja wejścia, obsługa błędów i stanu interfejsu | błędne dane dają czytelny komunikat; wyniki nie pozostają myląco po zmianie ustawień |
| P1 | Porównywanie modeli oraz wizualizacje portfela i korelacji | porównanie trzech modeli dla identycznej konfiguracji, bez interpretowania najwyższego VaR jako najgorszego modelu |
| P1 | Uporządkować eksport tabel i wykresów do rozdziału badawczego | w raporcie są metodologia, podział danych, testy VaR, Quantile Loss i ograniczenia |
| P2 | Przyspieszyć obliczenia przez cache parametrów GARCH i danych | powtarzane żądania nie estymują niepotrzebnie wszystkiego od nowa |
| P2 | Ewentualnie rozszerzyć o aktualne notowania i prognozy | nowy pipeline danych i osobna walidacja; nie mieszać z końcową oceną TEST |

**Przydział osób do powyższych zadań jest do uzgodnienia** — tabela nie zakłada konkretnego podziału ról w zespole.

## E. Co jest poza ZIP-em i może wymagać osobnego przekazania

`risk_module.zip` to minimalny pakiet do wykonywania prognoz. Nie zawiera wszystkich wcześniejszych skryptów badawczych, testów statystycznych, wykresów ani tabel porównawczych. Jeśli zespół chce odtworzyć **całą część eksperymentalną**, trzeba osobno dodać do repo uporządkowane skrypty i wyniki, m.in. `portfolio_final_test_daily.csv`, `portfolio_final_test_statistical.csv`, `portfolio_final_quantile_loss.csv` oraz odpowiednie skrypty generujące te pliki. Trzeba też zadbać o brak duplikatów i spójne nazwy katalogów.

**Znane ograniczenia techniczne:**
- Obliczenia są wykonywane z użyciem ścieżek względnych do `risk_module/`.
- Nie ma serwera HTTP; aplikacja Streamlit wywołuje funkcję Pythona bezpośrednio.
- Parametry GARCH dopasowywane są na TRAIN przy każdym wywołaniu API, co wpływa na wydajność.
- Pierwszy interfejs Streamlit nie był częścią ostatniego ZIP-u.
- Plik `requirements.txt` zawiera nazwy pakietów bez przypiętych wersji; dla pełnej odtwarzalności warto ustalić wersje zespołowe.

## F. Proponowana wiadomość przy przekazaniu

> Cześć, wrzuciłem moduł analizy ryzyka oparty na 10 ETF-ach. Zawiera przygotowane dane, modele GARCH i kopuł (Gauss, Student, Gauss z korelacją kroczącą), symulacje Monte Carlo, VaR/ES oraz interfejs `analyze_portfolio()` do integracji ze Streamlit. Całość testowałem na historycznym zbiorze TEST (2023–2026); w repo są instrukcje uruchomienia i przykład użycia. Szczegóły tego, co jest gotowe, wyniki badań oraz rzeczy do przejęcia rozpisałem w `HANDOFF_DLA_ZESPOLU.md`. Trzeba jeszcze podłączyć moduł do docelowej aplikacji, dopracować UI, testy integracyjne i finalne wykresy/raport. Uwaga: moduł obecnie obsługuje historyczne daty TEST, a nie bieżące notowania. Dajcie znać, kto przejmuje poszczególne zadania.
