# Backtest: próby, opóźnienie wykonania, stopa wolna od ryzyka, scenariusze

Dokument opisuje metodologię backtestu strategii opartych na modelach zależności. Stan: Etapy 0–2
(dane, próby, r_f, modele zależności, scenariusze adaptacyjne). Profile inwestora, optymalizacja,
testy ryzyka, MC ex post i ranking (Etapy 3–6) zostaną opisane po implementacji.

## Dane i próby

- Dzienne log-stopy w PLN (`data/processed/log_returns_pln.csv`) z `data_import.py` i
  `data_manipulation.py`, zakres deklarowany 2005-01-01 – 2026-10-29. AVB: Yahoo (ponowienie
  zapytania dla pojedynczego tickera), zapasowo Stooq `avb.us` (stooq.pl, stooq.com). Kontrola
  kompletności: `pf.data.quality` (`coverage.csv`, `log_returns_pln_check.json`).
- Próby (`config/backtest.yaml`): kalibracyjna 2008-01-01 – 2020-12-31 (wybór parametrów meta),
  hold-out 2021-01-01 – 2026-10-29 (ocena końcowa). Data rebalansowania należy do próby, w której
  leży; okres realizacji ostatniej daty kalibracji kończy się w hold-out.
- Pierwsza data rebalansowania wynika z `min_obs` (750 sesji w oknie 1260) i warunku, że każdy
  sub-fundusz ma aktywo z pełnym oknem (`covers_all_subfunds`). Sub-fundusz high yield (HYG od
  04.2007, JNK od 12.2007) przesuwa tę datę na ok. 2010-04, jeśli warunek nie zostanie złagodzony.

## Opóźnienie wykonania

Decyzja w sesji t (pierwsza sesja miesiąca) korzysta z danych ≤ t−1. Przy `execution_lag: L`
okres realizacji obejmuje sesje o indeksach [i(t) + L, i(t_next) + L):

| L | transakcja | okres realizacji |
|---|---|---|
| 1 (domyślnie) | zamknięcie t | (t, t_next] |
| 0 (wrażliwość, zachowanie `monthly_rebalance`) | zamknięcie t−1 | [t, t_next) |

Scenariusze ex ante obejmują H + L dni od końca okna; suma log-stóp liczona jest z dni L+1, …, L+H
(dni opóźnienia przesuwają stan GARCH, ale nie wchodzą do stopy okresu).

## Stopa wolna od ryzyka

`pf.data.riskfree`: WIBOR 1M ze Stooq (symbol `plopln1m`), dzienny szereg na dniach roboczych
z przeniesieniem ostatniego fixingu (≤ `max_ffill_days`); dni bez WIBOR (koniec szeregu przed końcem
próby, dłuższe luki) uzupełnia stopa referencyjna NBP (archiwum XML stóp NBP). Liczba i zakresy dni
uzupełnionych są zapisywane w manifeście. Stopa okresu (ACT/365):

  r_f(t0, t1) = WIBOR1M(t0) · (t1 − t0)[dni] / 365,

gdzie t0, t1 – zamknięcia sesji otwarcia i zamknięcia pozycji (`pf.backtest.data.holding_bounds`).
Wynik: `artifacts/market/riskfree_pln.parquet`.

## Wyniki i identyfikator przebiegu

Wyniki zapisywane są w `artifacts/backtest/<run_id>/`, gdzie run_id to pierwsze 12 znaków
SHA-256 kanonicznego JSON-a konfiguracji i skrótu pliku danych. Ciężkie pliki pośrednie
(marginesy i modele zależności per data – `artifacts/backtest/cache/`, ogony scenariuszy) są poza
repozytorium (`.gitignore`). Klucz pamięci podręcznej modeli to skrót danych okna estymacji
i konfiguracji modeli – zmiana danych poza oknem nie unieważnia modeli wcześniejszych dat.
`artifacts/rebalance/` zawiera wyniki historyczne (dane 2011–2020, lag 0, `itau`) i nie jest
nadpisywany.

## Modele zależności

Siedem modeli na wspólnych marginesach (opis i wzory: `docs/dependence_model.md`): `rvine`,
`cvine`, `dvine`, `gauss`, `student_t`, `indep`, `empirical`. Ocena na oknie estymacji (AIC, BIC,
test Vuonga z poprawką Schwarza dla R-vine vs pozostałe modele z gęstością) i poza próbą
(log-score kopuły na PIT z filtru GARCH naprzód, zapisywany per data i model; bez `empirical`).

## Scenariusze ex ante z adaptacyjną liczbą

`pf.sim.scenarios.simulate_adaptive`: partie po B = 2500 scenariuszy. Po każdej partii (od N ≥ 5000)
dla portfeli kontrolnych k – równe wagi w każdym sub-funduszu i równe wagi sub-funduszy – liczone są
straty L = −Rᵀw, VaR̂_α, ÊS_α (α = 0,975) i błąd standardowy

  SE(ÊS_α) = √( [Var(L | L > VaR_α) + α (ES_α − VaR_α)²] / ((1 − α) N) ).

Stop: max_k SE_k/|ÊS_k| ≤ ε (ε = 0,02; ostatnia data 0,01) albo N ≥ 50 000. N, ÊS i SE zapisywane są
per (data, model, portfel kontrolny). Pokrycie przedziałów ÊS ± 1,96·SE (2000 powtórzeń): N(0, 1),
N = 5000 – 95,2%; t₄, N = 5000 – 94,2%; t₄, N = 20 000 – 95,4%.

Wspólne liczby losowe: wejściowe W ~ U(0, 1) każdej partii pochodzą ze strumienia o ziarnie
(seed, data), wspólnego dla wszystkich modeli – vine'y, Gauss i niezależność przekształcają te same W;
losowość specyficzna dla modelu (χ² kopuły t, numer wiersza kopuły empirycznej) ma ziarno
(seed, data, CRC32(nazwa modelu)). Wszystkie strategie i profile danej daty korzystają z tych samych
scenariuszy danego modelu. Pierwsze N scenariuszy nie zależy od liczby partii.

## Ograniczenia (dotychczasowe)

- Notowania GPW i NYSE oraz fixing NBP nie są synchroniczne (zamknięcia 17:00 i 22:00, fixing
  ok. 11:50), więc dzienna zależność między rynkami jest zaniżona, a przeliczenie na PLN używa kursu
  z innej chwili niż cena.
- Horyzont scenariuszy H = 21 sesji jest stały, a okres realizacji ma 19–23 sesje.
