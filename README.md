# bsc-project – ryzyko i alokacja funduszu z uwzględnieniem zależności aktywów

Projekt bada, czy jawne modelowanie zależności między aktywami (kopuły winorośli i kopuły
eliptyczne) poprawia alokację wielosegmentowego funduszu dla inwestorów o różnej awersji do ryzyka.
Fundusz składa się z 9 sub-funduszy (79 instrumentów z GPW i giełd USA), a wszystkie wyniki są
liczone w PLN, ze stopą wolną od ryzyka WIBOR 1M.

Pakiet `pf` (`src/pf`) obejmuje:

- marginesy AR(1)–GJR-GARCH(1,1) ze skośnym rozkładem t Hansena;
- 7 modeli zależności: R-, C- i D-vine, kopuły Gaussa, t, niezależności i empiryczną;
- miesięczne scenariusze stóp zwrotu z adaptacyjną liczbą symulacji;
- dwupoziomową optymalizację składu według ekwiwalentu pewności CRRA z karą za zależność w dolnym
  ogonie, dla 5 profili inwestora;
- backtest kroczący 2008–2026 z próbą hold-out;
- testy wsteczne VaR/ES, MC ex post oraz ranking strategii i modeli.

Szczegóły metody i wzory: [`docs/backtest.md`](docs/backtest.md) i
[`docs/dependence_model.md`](docs/dependence_model.md).

## Najważniejsze wyniki

Wyniki pochodzą z przebiegu `4010467a06e7`: 226 dat rebalansowania 2008-01 – 2026-10, okres
kalibracji 156 miesięcy, hold-out 69 miesięcy (2021-01 – 2026-09). Wszystkie stopy są miesięczne,
netto po kosztach 0,1% wartości transakcji. Zrealizowany ekwiwalent pewności (CE) liczono od stóp
nadwyżkowych nad WIBOR 1M, w skali rocznej, przy γ danego profilu.

| hold-out, CE [% rocznie] | `ce_dep`* | `ce`* | R-vine × `ce_dep` | `ew` | `mv_hist` | `maxret_es`* | `min_es`* |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 konserwatywny (γ = 10) | 3,76 | 4,32 | 4,46 | 3,52 | 1,85 | 2,29 | −0,23 |
| P3 zrównoważony (γ = 5) | 6,87 | 7,03 | 6,82 | 5,26 | 5,18 | 5,06 | 0,81 |
| P5 agresywny (γ = 1,5) | 15,04 | 15,43 | 15,29 | 6,49 | 12,11 | 6,84 | 1,58 |

\* Średnia po 7 modelach zależności.

- **CE vs pozostałe reguły:** strategie maksymalizujące CE wygrywają w obu próbach z równymi wagami,
  średnią–wariancją i regułami ES.
- **Kara za zależność:** podnosi CE w kalibracji dla 4 z 5 profili, ale w hold-out nie.
- **Istotność:** różnice między strategiami CE w hold-out są statystycznie nierozróżnialne. Model
  Confidence Set obejmuje 24–27 z 30 szeregów w każdym profilu, a test Ledoita–Wolfa nie odrzuca
  równości wskaźnika Sharpe'a z `ew` dla strategii głównej.
- **Modele zależności:** R-vine ma najniższe BIC i wygrywa test Vuonga w każdym oknie estymacji.
  Według log-score poza próbą w hold-out równorzędne są kopuła Gaussa i R-vine.
- **Prognozy ryzyka:** w kalibracji prognozy VaR/ES 97,5% niedoszacowują ryzyka (11,4 przekroczenia
  wobec 3,9 oczekiwanych, głównie w 2008 r.). W hold-out jest 4,2 przekroczenia wobec 1,7
  oczekiwanych, a testy odrzucają rzadziej. Moc testów przy danych miesięcznych jest niska.

Pełne tabele: [`backtest_summary.md`](artifacts/reports/backtest_summary.md),
[`ranking_holdout.md`](artifacts/reports/ranking_holdout.md),
[`ranking_calibration.md`](artifacts/reports/ranking_calibration.md),
[`dependence_models.md`](artifacts/reports/dependence_models.md),
[`es_mc.md`](artifacts/reports/es_mc.md); wykresy w [`figures/backtest/`](figures/backtest/).

## Uniwersum i dane

| SF | segment | tickery |
|---|---|---|
| SF1 | akcje polskie | PKO, PZU, KGH, PKN, PEO, LPP, CDR, DNP, PGE, CPS (GPW) |
| SF2 | akcje USA | AAPL, MSFT, NVDA, AMZN, GOOGL, JPM, JNJ, XOM, PG, WMT |
| SF3 | rynki rozwinięte ex-USA | EFA, EWJ, EWU, EWG, EWQ, EWL, EWC, EWA, EWN, EWS |
| SF4 | rynki wschodzące | EEM, MCHI, INDA, EWZ, EWW, EWT, THD, IDX, TUR, EZA |
| SF5 | obligacje skarbowe USA | SHY, IEI, IEF, TLT, GOVT, VGSH, VGIT, VGLT, SCHO, SCHR |
| SF6 | obligacje korporacyjne IG | LQD, VCIT, VCSH, IGSB, IGIB, USIG |
| SF7 | obligacje high yield | HYG, JNK, ANGL, FALN |
| SF8 | złoto i spółki wydobywcze | GLD, IAU, SGOL, GDX, GDXJ, NEM, AEM, KGC, WPM, FNV |
| SF9 | nieruchomości (REIT) | PLD, AMT, EQIX, O, SPG, PSA, WELL, DLR, EXR (AVB niedostępny) |

**Notowania.** Dzienne notowania z okresu 2005-01-04 – 2026-10-29 pochodzą z Yahoo Finance, a jako
źródło zapasowe służy Stooq. Ceny w USD są przeliczane na PLN po średnim kursie NBP. Dni bez obrotu
(zerowy wolumen) są pomijane. Wynik: dzienne log-stopy `data/processed/log_returns_pln.csv`.

**Brak AVB.** AvalonBay wycofano z obrotu w 2026 r. i jego historia nie jest już dostępna w Yahoo
ani w Nasdaq.

**Stopa wolna od ryzyka.** Docelowo dzienny WIBOR 1M (Stooq). Ponieważ ten był niedostępny, użyto
średniej miesięcznej WIBOR 1M z Eurostatu, a dla 10.2026 stopy referencyjnej NBP. Wynik:
`artifacts/market/riskfree_pln.parquet`.

**Skład sub-funduszy w czasie.** Instrumenty notowane krócej niż 750 sesji w oknie nie wchodzą do
optymalizacji. Sub-fundusz bez takiego instrumentu ma wagę 0: SF7 do 2010-05, SF1 do 2008-02.

## Metoda w skrócie

1. **Marginesy.** Dla każdego instrumentu AR(1)–GJR-GARCH(1,1) ze skośnym rozkładem t, w oknie
   1260 sesji przed datą decyzji. Panel standaryzowanych reszt jest przekształcany w
   pseudo-obserwacje z rang.
2. **Modele zależności.** Siedem kopuł estymowanych na tych samych pseudo-obserwacjach. Winorośle
   mają obcięcie na 3 drzewach i sekwencyjną MNW par z rodzin Gauss/t/Clayton/Gumbel (z obrotami)
   i Frank, z wyborem rodziny według AIC. Ocena: AIC/BIC, test Vuonga i log-score poza próbą.
3. **Scenariusze ex ante.** 21-dniowe ścieżki GARCH z kopuły z adaptacyjną liczbą N: symulacja
   kończy się, gdy błąd standardowy ES jest ≤ 2% jego wartości. Wszystkie modele korzystają ze
   wspólnych liczb losowych.
4. **Optymalizacja dla każdego profilu** (P1–P5 różnią się γ, κ i limitami):

   max CE_γ(w) − κ·σ^EW·wᵀΛ(q)w,

   gdzie CE_γ to ekwiwalent pewności CRRA na scenariuszach, a Λ(q) to macierz zależności w dolnym
   ogonie. Optymalizacja ma dwa poziomy: najpierw wagi tickerów w sub-funduszu, potem wagi
   sub-funduszy. Obowiązują limity wagi tickera w SF, wagi SF i ekspozycji look-through.
   Porównywane strategie: `ce_dep`, `ce` (κ = 0), `maxret_es`, `min_es` oraz benchmarki `ew`
   i `mv_hist` (shrinkage Ledoita–Wolfa).
5. **Backtest.** Rebalansowanie co miesiąc. Decyzja zapada w pierwszej sesji miesiąca na danych do
   dnia poprzedniego, a transakcja na jej zamknięciu. Między datami portfel jest trzymany z dryfem
   wag. Analizę wrażliwości (opóźnienie 0, q = 0,10, ρ_S, siatka κ) liczono tylko na kalibracji.
   Parametry zamrożono przed hold-out.
6. **Ocena.** Testy Kupca, Christoffersena, Acerbiego–Szekely (Z₂) i Du–Escanciano. MC ex post: FHS,
   bootstrap stacjonarny i wielowymiarowe MC. Ranking: CE, Sharpe z SE Lo, Sortino, Calmar, Omega,
   STARR, MCS i test Ledoita–Wolfa różnicy Sharpe'a.

## Struktura repozytorium

```
config/            parametry (YAML): models, backtest, profiles, subfunds, options
data/              dane surowe (raw/) i przetworzone (processed/); data_import.py, data_manipulation.py
src/pf/
  data/            pobieranie danych, stopa wolna od ryzyka (riskfree), kontrola kompletności (quality)
  models/          marginesy GARCH, kopuły parowe, R-vine, interfejs modeli zależności, ocena (gof)
  sim/             scenariusze (adaptacyjne N, wspólne liczby losowe)
  opt/             CE i gradient (utility), kara za zależność, profile i optymalizacja dwupoziomowa,
                   LP Rockafellara–Uryaseva (cvar), średnia–wariancja z Ledoitem–Wolfem (mv)
  backtest/        daty i okna (data), krok daty (engine), realizacja stóp (realized), testy VaR/ES,
                   miary, MC ex post, ranking
  options/         moduł opcji (wycena, IV, powierzchnia zmienności, PPUT)
pipelines/         skrypty uruchomieniowe (python -m pipelines.<nazwa>)
artifacts/         wyniki: backtest/<run_id>/ (tabele), reports/ (raporty .md), market/, models/
figures/           wykresy (log_returns/, backtest/)
docs/              dokumentacja metodologii
tests/             testy jednostkowe i integracyjne (bez sieci)
```

## Instalacja i testy

```bash
pip install -e ".[dev]"
pytest -q              # ok. 200 testów bez dostępu do sieci (ok. 1 min)
pytest -m network      # testy sieciowe (źródła danych)
```

CI (`.github/workflows/ci.yml`) uruchamia `pytest -q` dla każdego pusha i PR.

## Uruchomienie

Pipeline'y uruchamia się w podanej kolejności. Są przyrostowe i wznawialne: policzone daty i modele
są pomijane.

```bash
# 1. Dane (sieć): notowania, kursy NBP, log-stopy w PLN, stopa wolna od ryzyka
python data_import.py
python data_manipulation.py
python -m pf.data.riskfree

# 2. Modele zależności na próbie kalibracyjnej (ok. 3 min) → artifacts/reports/dependence_models_full_sample.md
python -m pipelines.fit_rvine_full_sample --jobs 4

# 3. (opcjonalnie) pomiar kosztu i prognoza czasu backtestu → artifacts/reports/scenario_timing.md
python -m pipelines.scenario_timing --jobs 3

# 4. Backtest (ok. 42 h CPU; na 4 rdzeniach ok. 11,5 h): kalibracja, zamrożenie parametrów, hold-out
python -m pipelines.backtest --sample calibration --jobs 4
python -m pipelines.backtest --sample holdout --jobs 4
python -m pipelines.backtest --consolidate-only

# 5. MC ex post (ok. 10 min) i ranking z raportami i wykresami (ok. 1 min)
python -m pipelines.expost_mc --jobs 4
python -m pipelines.ranking
```

Pojedyncze daty: `python -m pipelines.backtest --dates 2008-10-01 --no-consolidate`.

Starszy pipeline `python -m pipelines.monthly_rebalance` (skład sub-funduszy na danych 2011–2020,
wyniki w `artifacts/rebalance/`) pozostaje bez zmian.

## Wyniki i artefakty

| plik | zawartość |
|---|---|
| `artifacts/reports/backtest_summary.md` | podsumowanie przebiegu, najlepsze szeregi, testy VaR/ES, wrażliwość |
| `artifacts/reports/ranking_{calibration,holdout}.md` | ranking strategii per profil (CE, Sharpe, MCS, p-wartości testów) |
| `artifacts/reports/dependence_models.md` | ranking modeli zależności (AIC/BIC, Vuong, log-score, testy) |
| `artifacts/reports/es_mc.md` | VaR/ES ex post (FHS, bootstrap, MC wielowymiarowe) vs ex ante i historyczne |
| `artifacts/reports/weights/` | wagi tickerów strategii głównej w czasie (CSV, opis w README) |
| `artifacts/backtest/<run_id>/periods.parquet` | miesięczne stopy brutto/netto, koszty, obrót, r_f każdego szeregu |
| `artifacts/backtest/<run_id>/weights_tickers.parquet` | wagi tickerów, SF i look-through wszystkich 146 szeregów |
| `artifacts/backtest/<run_id>/forecasts.parquet` | prognozy ex ante (VaR, ES, E[R], CE) z realizacją i PIT |
| `artifacts/backtest/<run_id>/tests.parquet` | testy Kupca, Christoffersena, Z₂, Du–Escanciano |
| `figures/backtest/` | majątek i obsunięcia, skład W w czasie, krocząca Sharpe, ES vs straty |

Szereg identyfikuje czwórka `model | strategia | profil | wariant`, np. `rvine|ce_dep|P3|base`.
Benchmarki mają model `-`.

## Odtwarzalność

- **Identyfikator przebiegu.** `run_id` to skrót SHA-256 konfiguracji i pliku danych. Zmiana
  parametrów tworzy nowy katalog `artifacts/backtest/<run_id>/`.
- **Ziarna losowe.** Zależą od daty (wspólne W dla modeli) oraz od daty i modelu.
- **Pamięć podręczna.** Modele z danej daty są zapisywane w `artifacts/backtest/cache/` pod
  skrótem danych okna, więc dopisanie nowych sesji nie wymusza ponownej estymacji wcześniejszych dat.
- **Manifest.** `artifacts/manifest.json` przechowuje skróty plików, wersje pakietów i parametry.
- **Parametry hold-out.** Zamrożone parametry meta zapisuje `frozen_params.json`.
- **Pliki poza repozytorium.** Ciężkie pliki pośrednie (`dates/`, `tails/`, stopy dzienne) są
  w `.gitignore`; odtwarza je ponowne uruchomienie pipeline'u.

## Ograniczenia

- **Asynchroniczność.** Notowania GPW i NYSE oraz fixing NBP nie są synchroniczne, przez co dzienna
  zależność między rynkami jest zaniżona.
- **Heurystyka dwupoziomowa.** Dekompozycja optymalizuje CE pojedynczego SF z γ inwestora, więc nie
  daje optimum łącznego.
- **Oczekiwane stopy.** Pochodzą ze stałych AR(1) w oknie i są obarczone dużym błędem.
- **Winsoryzacja innowacji.** Symulowane innowacje są ograniczone do 2·max|ε̂| z okna, co zapobiega
  eksplozji rekursji GARCH przy rozkładach z η < 4.
- **Krótki hold-out.** 69 miesięcy i niska moc testów miesięcznych utrudniają rozstrzygnięcia.

Pełna lista: [`docs/backtest.md`](docs/backtest.md), sekcja 11.

## Moduł opcji

Osobny moduł (`src/pf/options`, `docs/options_module.md`) obejmuje dane zmienności (indeksy Cboe),
zrzuty łańcuchów opcji, wycenę Blacka–Scholesa–Mertona, zmienność implikowaną, parametryczną
powierzchnię zmienności i replikację strategii PPUT. Dane rynkowe są odświeżane codziennie przez
`.github/workflows/refresh.yml`. Kalibracja i walidacja:
`python -m pipelines.calibrate_surface`, `python -m pipelines.validate_pput`.
