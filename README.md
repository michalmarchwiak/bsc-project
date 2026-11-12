# bsc-project – ryzyko i alokacja funduszu z uwzględnieniem zależności aktywów

Projekt bada, czy jawne modelowanie zależności między aktywami (kopuły winorośli i kopuły
eliptyczne) poprawia alokację wielosegmentowego funduszu dla inwestorów o różnej awersji do ryzyka.
Fundusz składa się z 10 sub-funduszy: ok. 80 instrumentów z GPW i giełd USA w każdej dacie (SF1–SF9; spółki
wybierane punktowo w czasie z puli ok. 260 instrumentów) oraz 8 rodzin obligacji oszczędnościowych Skarbu
Państwa (SF10). Wszystkie wyniki są liczone w PLN, ze stopą wolną od ryzyka WIBOR 1M.

Pakiet `pf` (`src/pf`) obejmuje:

- marginesy AR(1)–GJR-GARCH(1,1) ze skośnym rozkładem t Hansena oraz margines deterministyczny
  dla obligacji oszczędnościowych (stopa znana ex ante);
- 7 modeli zależności: R-, C- i D-vine, kopuły Gaussa, t, niezależności i empiryczną;
- miesięczne scenariusze stóp zwrotu z adaptacyjną liczbą symulacji;
- dwupoziomową optymalizację składu według ekwiwalentu pewności CRRA z karą za zależność w dolnym
  ogonie, dla 5 profili inwestora;
- backtest kroczący 2008–2026 z próbą hold-out;
- testy wsteczne VaR/ES, MC ex post oraz ranking strategii i modeli.

Szczegóły metody i wzory: [`docs/backtest.md`](docs/backtest.md),
[`docs/dependence_model.md`](docs/dependence_model.md) i [`docs/savings_bonds.md`](docs/savings_bonds.md) (SF10).

## Najważniejsze wyniki

**Uwaga.** Poniższe wyniki pochodzą z przebiegu `3ababedef5d8`, policzonego przed dwiema zmianami konfiguracji:
niższym limitem wagi sub-funduszu w P1–P3 (`w_max` = 0,20 / 0,225 / 0,25 zamiast 0,40 / 0,45 / 0,50) i górnym
ograniczeniem mnożnika m_t w nakładce opcyjnej (`m_cap`). Wyniki dla bieżącej konfiguracji wymagają nowego
przebiegu (kroki 4–6 w sekcji „Uruchomienie”).

Przebieg `3ababedef5d8`: uniwersum `pit`, kopuły na sumach 5-dniowych reszt, 10 sub-funduszy z SF10, nakładka
opcyjna `_ov`, benchmarki rynkowe. 226 dat rebalansowania 2008-01 – 2026-10, okres kalibracji 156 miesięcy,
hold-out 69 miesięcy (2021-01 – 2026-09), 232 szeregi bazowe, 45 h CPU. Wszystkie stopy są miesięczne, netto po
kosztach 0,1% wartości transakcji. Zrealizowany ekwiwalent pewności (CE) liczono od stóp nadwyżkowych nad
WIBOR 1M, w skali rocznej, przy γ danego profilu.

CE [% rocznie]: hold-out, w nawiasie kalibracja.

| profil | `ce_dep`* | `ce`* | `ce_dep_ov`* | `maxret_es`* | `min_es`* | `ew` | `mv_hist` | `idx_spx` | `idx_wig20` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 (γ = 10) | 2,16 (0,71) | 1,03 (0,10) | – | 1,33 (−1,01) | 1,42 (−0,39) | 3,08 (0,81) | 2,02 (0,16) | 2,49 (−1,22) | −9,54 (−25,59) |
| P2 (γ = 7) | 3,06 (2,55) | 2,09 (1,91) | – | 2,51 (0,89) | 2,01 (0,56) | 4,09 (2,50) | 3,36 (0,66) | 5,20 (2,54) | −1,94 (−17,76) |
| P3 (γ = 5) | 3,59 (4,29) | 2,52 (4,73) | 4,37 (5,86) | 3,82 (2,44) | 2,35 (1,15) | 4,77 (3,64) | 5,62 (2,49) | 7,03 (5,03) | 2,99 (−12,83) |
| P4 (γ = 3) | 5,88 (8,57) | 6,92 (8,59) | 7,03 (9,90) | 5,82 (4,42) | 2,74 (1,62) | 5,45 (4,80) | 8,43 (5,58) | 8,88 (7,50) | 7,81 (−8,04) |
| P5 (γ = 1,5) | 13,02 (14,26) | 13,12 (14,66) | 14,79 (15,74) | 7,98 (5,59) | 3,00 (1,90) | 5,96 (5,69) | 11,42 (5,92) | 10,28 (9,36) | 11,35 (−4,49) |

\* Średnia po 7 modelach zależności. Warianty `_ov` (nakładka opcyjna, polityka P3) liczono dla profili P3–P5.
Strategia główna R-vine × `ce_dep`: 2,53 / 3,85 / 3,30 / 3,48 / 10,92 w hold-out (P1–P5).

- **CE vs benchmarki:** przewaga strategii CE ogranicza się do profili agresywnych. W hold-out `ce` i `ce_dep`
  wygrywają ze wszystkimi benchmarkami tylko w P5. W P1–P3 lepsze są równe wagi (`ew`) i S&P 500 (`idx_spx`),
  a w P4 `idx_spx`, `mv_hist` i `idx_wig20`. W kalibracji strategie CE wygrywają w P4–P5, w P3 przegrywają
  z `idx_spx`, w P2 są na poziomie `ew` i `idx_spx`, a w P1 przegrywają z `ew`.
- **Wpływ uniwersum:** w przebiegu `83254f06e339` (stała lista spółek dobrana ex post, panel dzienny) strategie CE
  wygrywały w hold-out z `ew` i `mv_hist` we wszystkich profilach (`ce_dep`: 4,53–16,10). Różnica łączy zmianę
  uniwersum i agregacji reszt; tych efektów nie rozdzielono.
- **Kara za zależność:** `ce_dep` ma wyższy CE niż `ce` w P1–P2 w obu próbach, w P3 tylko w hold-out, a w P4–P5
  niższy. Na siatce κ (kalibracja) najlepsze jest κ = 2 dla P1–P3 i κ ≤ 0,05 dla P4–P5.
- **Koncentracja:** `ce_dep` trzyma średnio 6,6 (P1) do 2,5 (P5) sub-funduszy, a SF1 i SF2 mają łącznie 46% (P1)
  do 88% (P5) wagi w hold-out. Limit `w_max` wiąże w mniej niż 5% dat. Wagę SF ogranicza limit look-through:
  W_k ≤ c_LT / maxᵢ w_{k,i}, co daje ok. 0,25 (P1) do 0,50 (P5). Po tym przebiegu `w_max` w P1–P3 obniżono
  poniżej tego progu.
- **Nakładka opcyjna:** warianty `ce_ov`, `ce_dep_ov` i `maxret_es_ov` mają wyższy średni CE niż bazowe
  (hold-out: +0,8 do +1,8 p.p., kalibracja: +1,2 do +3,1 p.p.). Różnica skupia się w dwóch miesiącach
  (2020-09 i 2021-01), w których modelowa σ_imp jest 2,5 razy wyższa od prognozy GARCH (m_t ≈ 2,5 przy
  medianie 1,08), a cały portfel trafia do kolumn SFk_OV. Bez nich przewaga w hold-out spada do +0,1 do
  +0,4 p.p., a w kalibracji do +0,4 do +2,4 p.p. `min_es_ov` ma CE ujemny. Wynik zależy więc od założeń
  wyceny (sekcja „Moduł opcji”). Po tym przebiegu dodano ograniczenie `m_cap`.
- **Istotność:** różnice są statystycznie nierozróżnialne. Model Confidence Set w hold-out obejmuje wszystkie
  32 szeregi w P1–P2 oraz 49–59 z 60 szeregów w P3–P5. Test Ledoita–Wolfa nie odrzuca równości wskaźnika
  Sharpe'a z `ew` dla strategii głównej (p = 0,27–0,97).
- **Modele zależności:** R-vine ma najniższe BIC, wygrywa test Vuonga w każdym oknie kalibracji i jako jedyny
  należy do MCS według log-score poza próbą w obu próbach. Lepsze dopasowanie nie daje lepszych portfeli:
  najlepszą średnią rangę CE ma C-vine, a w hold-out najlepsze szeregi modelowe dają kopuły Gaussa i t.
- **Prognozy ryzyka:** prognozy VaR/ES 97,5% niedoszacowują ryzyka. W kalibracji jest średnio 14,7 przekroczenia
  wobec 3,9 oczekiwanych (26% w 2008 r., dalsze skupienia w 2015 i 2018 r.), w hold-out 6,1 wobec 1,7
  (55% w 2022 r.). Test Kupca nie odrzuca dla 42% szeregów w kalibracji i 61% w hold-out. Moc testów przy
  danych miesięcznych jest niska.

Pełne tabele: [`backtest_summary.md`](artifacts/reports/backtest_summary.md),
[`ranking_holdout.md`](artifacts/reports/ranking_holdout.md),
[`ranking_calibration.md`](artifacts/reports/ranking_calibration.md),
[`dependence_models.md`](artifacts/reports/dependence_models.md),
[`es_mc.md`](artifacts/reports/es_mc.md); wykresy w [`figures/backtest/`](figures/backtest/).
Sekcja 9 w [`docs/backtest.md`](docs/backtest.md) opisuje jeszcze przebieg `83254f06e339`.

## Uniwersum i dane

`config/subfunds.yaml` definiuje dwa uniwersa (klucz `universe` albo zmienna `PF_UNIVERSE`):

- **`pit` (domyślne)** – skład SF1, SF2, SF9 i spółek wydobywczych w SF8 jest wybierany punktowo w czasie z szerszej
  puli spółek (ok. 40 spółek GPW, ok. 120 dużych spółek USA, 36 REIT-ów, 18 spółek wydobywczych). W dacie t do SF
  wchodzi 10 spółek (SF8: 5 spółek obok 5 ETF-ów na złoto) o największej wartości obrotu z 12 miesięcy przed
  ostatnią rewizją styczniową, spośród spółek z historią ≥ 750 sesji. Reguła używa tylko danych sprzed t
  (`pf.universe`, opis: [`docs/backtest.md`](docs/backtest.md)). Modele daty obejmują ok. 80 aktywów składu.
- **`fixed`** – pierwotna stała lista z tabeli poniżej, dobrana z perspektywy 2026 r. (obciążenie doborem ex post);
  zachowana do porównań.

Benchmarki rynkowe (`market_benchmarks` w `config/backtest.yaml`): S&P 500 z dywidendami (ETF SPY w PLN) i WIG20TR.

| SF | segment | tickery (uniwersum `fixed`) |
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
| SF10 | obligacje oszczędnościowe Skarbu Państwa | OTS, DOS, DOR, ROR, TOZ, TOS, COI, EDO (`*.SP`) |

**Notowania.** Dzienne notowania z okresu 2005-01-04 – 2026-10-29 pochodzą z Yahoo Finance, a jako
źródło zapasowe służy Stooq. Ceny w USD są przeliczane na PLN po średnim kursie NBP. Dni bez obrotu
(zerowy wolumen) są pomijane. Wynik: dzienne log-stopy `data/processed/log_returns_pln.csv`.

**Obligacje oszczędnościowe (SF10).** Nie są notowane, więc ich stopy wynikają z warunków emisji ze zbioru
obrotomierz.pl „Obligacje oszczędnościowe Skarbu Państwa: emisje i oprocentowanie” (Błażewski 2026,
[10.5281/zenodo.23208023](https://doi.org/10.5281/zenodo.23208023), CC BY 4.0; dane MF i listy emisyjne).
Indeks rodziny to wartość pozycji (nominał + narosłe odsetki) kupionej w 2005 r. lub przy debiucie
rodziny i rolowanej przy wykupie po cenie zamiany (`pf.data.savings_bonds`). Kupony są reinwestowane.
Sprzedaż przed wykupem obciąża opłata za wcześniejszy wykup z listu emisyjnego. Stopa w horyzoncie miesiąca
jest znana ex ante, więc SF10 ma margines deterministyczny i nie wchodzi do kopuł. Indeksy DOS i TOZ kończą
się po wycofaniu tych rodzin ze sprzedaży (2023-01, 2024-05). ROS i ROD (tylko „Rodzina 800+”) pominięto.
Szczegóły: [`docs/savings_bonds.md`](docs/savings_bonds.md).

**Brak AVB.** AvalonBay wycofano z obrotu w 2026 r. i jego historia nie jest już dostępna w Yahoo
ani w Nasdaq.

**Stopa wolna od ryzyka.** Docelowo dzienny WIBOR 1M (Stooq). Ponieważ ten był niedostępny, użyto
średniej miesięcznej WIBOR 1M z Eurostatu, a dla 10.2026 stopy referencyjnej NBP. Wynik:
`artifacts/market/riskfree_pln.parquet`.

**Skład sub-funduszy w czasie.** Instrumenty notowane krócej niż 750 sesji w oknie nie wchodzą do
optymalizacji. Sub-fundusz bez takiego instrumentu ma wagę 0: SF7 do 2010-05, SF1 do 2008-02.
Obligacje SF10 wymagają 21 sesji w oknie i notowania w sesji t−1 (OTS od 2017-11, ROR i DOR od 2022-07,
TOS od 2022-09; DOS do 2023-01, TOZ do 2024-05).

## Metoda w skrócie

1. **Marginesy.** Dla każdego instrumentu AR(1)–GJR-GARCH(1,1) ze skośnym rozkładem t, w oknie
   1260 sesji przed datą decyzji. Panel standaryzowanych reszt jest przekształcany w
   pseudo-obserwacje z rang. Obligacje SF10 mają margines deterministyczny: dzienna stopa w horyzoncie
   to średnia z 21 ostatnich sesji, bez innowacji. Nie wchodzą do panelu reszt ani do kopuł.
2. **Modele zależności.** Siedem kopuł estymowanych na tych samych pseudo-obserwacjach: rangach sum reszt
   z k = 5 kolejnych sesji (`dependence.aggregation`; k = 1 – panel dzienny). Sumy ograniczają wpływ
   asynchroniczności zamknięć GPW i NYSE oraz fixingu NBP na zależność w horyzoncie miesiąca. Winorośle
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
   Stała stopa SF10 w scenariuszach daje Λ_ij(q) = 0, więc SF10 nie zwiększa kary za zależność.
   Porównywane strategie: `ce_dep`, `ce` (κ = 0), `maxret_es`, `min_es` oraz benchmarki `ew`
   i `mv_hist` (shrinkage Ledoita–Wolfa).
5. **Backtest.** Rebalansowanie co miesiąc. Decyzja zapada w pierwszej sesji miesiąca na danych do
   dnia poprzedniego, a transakcja na jej zamknięciu. Między datami portfel jest trzymany z dryfem
   wag. Koszt: 0,1% wartości transakcji oraz, przy sprzedaży obligacji SF10, opłata za wcześniejszy
   wykup. Analizę wrażliwości (opóźnienie 0, q = 0,10, ρ_S, siatka κ) liczono tylko na kalibracji.
   Parametry zamrożono przed hold-out.
6. **Ocena.** Testy Kupca, Christoffersena, Acerbiego–Szekely (Z₂) i Du–Escanciano. MC ex post: FHS,
   bootstrap stacjonarny i wielowymiarowe MC. Ranking: CE, Sharpe z SE Lo, Sortino, Calmar, Omega,
   STARR, MCS i test Ledoita–Wolfa różnicy Sharpe'a.

## Struktura repozytorium

```
config/            parametry (YAML): models, backtest, profiles, subfunds, savings_bonds, options
data/              dane surowe (raw/) i przetworzone (processed/); data_import.py, data_manipulation.py
src/pf/
  data/            pobieranie danych, stopa wolna od ryzyka (riskfree), kontrola kompletności (quality),
                   indeksy obligacji oszczędnościowych SF10 (savings_bonds)
  models/          marginesy GARCH, kopuły parowe, R-vine, interfejs modeli zależności, ocena (gof)
  sim/             scenariusze (adaptacyjne N, wspólne liczby losowe)
  opt/             CE i gradient (utility), kara za zależność, profile i optymalizacja dwupoziomowa,
                   LP Rockafellara–Uryaseva (cvar), średnia–wariancja z Ledoitem–Wolfem (mv)
  universe.py      skład sub-funduszy punktowo w czasie (wybór spółek z puli według obrotów)
  backtest/        daty i okna (data), krok daty (engine), realizacja stóp (realized), testy VaR/ES,
                   miary, MC ex post, ranking
  options/         moduł opcji (wycena, IV, powierzchnia zmienności, PPUT) i miesięczna nakładka opcyjna
                   na tickery SF (rejestr, σ_imp, NAV, polityki P0–P3, warianty SFk_OV, walidacja BXM/PPUT)
pipelines/         skrypty uruchomieniowe (python -m pipelines.<nazwa>)
artifacts/         wyniki: backtest/<run_id>/ (tabele), reports/ (raporty .md), market/, models/
figures/           wykresy (log_returns/, backtest/)
docs/              dokumentacja metodologii
tests/             testy jednostkowe i integracyjne (bez sieci)
```

## Instalacja i testy

```bash
pip install -e ".[dev]"
pytest -q              # ok. 300 testów bez dostępu do sieci (ok. 2 min)
pytest -m network      # testy sieciowe (źródła danych)
```

CI (`.github/workflows/ci.yml`) uruchamia `pytest -q` dla każdego pusha i PR.

## Uruchomienie

Pipeline'y uruchamia się z katalogu repozytorium w podanej kolejności. Instalacja pakietu (`pip install -e .`) nie jest
wymagana: przy jej braku `pipelines/__init__.py` dopisuje `src/` do ścieżki importu. Są przyrostowe i wznawialne: policzone daty i modele
są pomijane.

```bash
# 1. Dane (sieć): notowania, kursy NBP, obligacje oszczędnościowe, log-stopy w PLN, obroty, stopa wolna od ryzyka
python data_import.py --missing-only   # tylko tickery bez pliku w data/raw/ohlcv (bez flagi: wszystkie od nowa)
python data_manipulation.py --no-plots # tabele stóp i data/processed/turnover_monthly.csv (bez flagi: także wykresy)
python -m pf.data.riskfree
# WIG20TR (benchmark) pochodzi ze Stooq; gdy pobranie się nie uda: zapisz CSV jako data/raw/stooq/wig20tr.csv

# 2. Modele zależności na próbie kalibracyjnej (ok. 3 min) → artifacts/reports/dependence_models_full_sample.md
python -m pipelines.fit_rvine_full_sample --jobs 4

# 3. (opcjonalnie) pomiar kosztu i prognoza czasu backtestu → artifacts/reports/scenario_timing.md
python -m pipelines.scenario_timing --jobs 3
#    (opcjonalnie) dobór agregacji panelu reszt: zależność stóp 21-dniowych w scenariuszach a w danych
python -m pipelines.dependence_diagnostics --jobs 4      # → artifacts/reports/dependence_diagnostics.md

# 4. Nakładka opcyjna (ok. 45 min) – pozycje opcyjne wymagane przez backtest (config/backtest.yaml: options_overlay)
python -m pipelines.options_monthly --jobs 4

# 5. Backtest (ok. 45 h CPU z wariantami _ov; na 4 rdzeniach ok. 11,5 h): kalibracja, zamrożenie parametrów, hold-out
python -m pipelines.backtest --sample calibration --jobs 4
python -m pipelines.backtest --sample holdout --jobs 4
python -m pipelines.backtest --consolidate-only

# 6. MC ex post (ok. 10 min) i ranking z raportami i wykresami (ok. 1 min)
python -m pipelines.expost_mc --jobs 4
python -m pipelines.ranking
```

Pojedyncze daty: `python -m pipelines.backtest --dates 2008-10-01 --no-consolidate`.
Przebieg na pierwotnej liście spółek: poprzedź każde polecenie z kroków 2–6 zmienną `PF_UNIVERSE=fixed`
(osobny `run_id`). Panel dzienny: `dependence.aggregation.days: 1` w `config/models.yaml`.

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
| `artifacts/backtest/<run_id>/weights_tickers.parquet` | wagi tickerów, SF i look-through wszystkich 230 szeregów bazowych |
| `artifacts/backtest/<run_id>/forecasts.parquet` | prognozy ex ante (VaR, ES, E[R], CE) z realizacją i PIT |
| `artifacts/backtest/<run_id>/tests.parquet` | testy Kupca, Christoffersena, Z₂, Du–Escanciano |
| `artifacts/backtest/<run_id>/universe.parquet` | skład uniwersum `pit` w każdej dacie (ranking obrotów, wybór spółek) |
| `figures/backtest/` | majątek i obsunięcia, skład W w czasie, krocząca Sharpe, ES vs straty |

Szereg identyfikuje czwórka `model | strategia | profil | wariant`, np. `rvine|ce_dep|P3|base`.
Benchmarki mają model `-`. Benchmarki rynkowe `idx_spx` i `idx_wig20` nie mają wag ani prognoz ex ante.

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

- **Asynchroniczność.** Notowania GPW i NYSE oraz fixing NBP nie są synchroniczne, przez co dzienna zależność
  jest zniekształcona. Kopuły na sumach 5-dniowych reszt ograniczają ten efekt, ale go nie usuwają.
- **Uniwersum.** Pule spółek w uniwersum `pit` obejmują tylko spółki notowane do dziś, a ranking opiera się na
  wartości obrotu (przybliżenie wielkości spółki), więc obciążenie przeżywalnością jest ograniczone, nie usunięte.
- **Heurystyka dwupoziomowa.** Dekompozycja optymalizuje CE pojedynczego SF z γ inwestora, więc nie
  daje optimum łącznego.
- **Oczekiwane stopy.** Pochodzą ze stałych AR(1) w oknie i są obarczone dużym błędem.
- **Winsoryzacja innowacji.** Symulowane innowacje są ograniczone do 2·max|ε̂| z okna, co zapobiega
  eksplozji rekursji GARCH przy rozkładach z η < 4.
- **Krótki hold-out.** 69 miesięcy i niska moc testów miesięcznych utrudniają rozstrzygnięcia.
- **Limity koncentracji.** W P4–P5 limit `w_max` nie wiąże. Wagę SF ogranicza iloraz c_LT / maxᵢ w_{k,i},
  więc profile agresywne trzymają 2–3 sub-fundusze, głównie SF1 i SF2. W P1–P3 `w_max` (0,20–0,25) wiąże
  z założenia. Wartości te wskazał przybliżony wariant kontrfaktyczny na przebiegu `3ababedef5d8`, w którym
  hold-out był już znany, więc ocena w hold-out ma charakter opisowy.
- **Nakładka opcyjna.** Mnożnik m_t = VIX/σ̂_SPX jest zaszumiony i bez ograniczenia sięga 2,5. Ograniczenie
  `m_cap` (kwantyl 95% historii m_s z okna estymacji) obniża go w 19 z 226 dat, najwyżej do 1,67. Premie
  tickerów proxy pozostają modelowe.
- **Indeksy obligacji SF10.** Odzwierciedlają rolowaną pozycję jednego rocznika, a nie serię bieżącą,
  którą kupuje inwestor w dniu t. Prognoza stopy z 21 sesji pomija zmianę stopy w nowym okresie odsetkowym.

Pełna lista: [`docs/backtest.md`](docs/backtest.md), sekcja 11.

## Moduł opcji

Osobny moduł (`src/pf/options`, `docs/options_module.md`) obejmuje dane zmienności (indeksy Cboe),
zrzuty łańcuchów opcji, wycenę Blacka–Scholesa–Mertona, zmienność implikowaną, parametryczną
powierzchnię zmienności i replikację strategii PPUT. Dane rynkowe są odświeżane codziennie przez
`.github/workflows/refresh.yml`. Kalibracja i walidacja:
`python -m pipelines.calibrate_surface`, `python -m pipelines.validate_pput`.

### Miesięczna nakładka opcyjna

`pipelines.options_monthly` wycenia w każdej dacie rebalansowania hipotetyczne opcje europejskie bliskie ATM
(K/S ∈ {0,95; …; 1,05}, |ln(K/F)| ≤ 0,06) na każdy ticker SF1–SF9. Opcje są rozliczane w kolejnej dacie
i włączane do NAV sub-funduszu według polityki: P0 (brak), P1 (stały put lub covered call), P2 (reguły na
cenie ubezpieczenia m_t, trendzie i reżimie zmienności) albo P3 (LP max E[R] − λ·CVaR na scenariuszach
GARCH-kopuła). σ_imp jest odtwarzana modelowo (GVZ dla ETF-ów na złoto, w pozostałych przypadkach
σ̂_GARCH × m_t, m_t = VIX/σ̂_SPX z górnym ograniczeniem `m_cap`), bo dla większości tickerów nie ma notowanych
opcji miesięcznych. Warianty SFk_OV wchodzą
jako kolumny poziomu 2 optymalizacji profili ≥ P3. Wycena jest walidowana replikacją Cboe BXM i PPUT.

```bash
python -m pipelines.options_monthly --jobs 4     # rejestr, polityki, NAV, portfele, walidacja, raport
```

Główny backtest uwzględnia nakładkę przy każdej reewaluacji (sekcja `options_overlay` w `config/backtest.yaml`):
dla profili ≥ P3 każda strategia ma bliźniaka `<strategia>_ov` z kolumnami SFk_OV na poziomie 2, a kurs USD/PLN
jest aktywem modelu. Wyniki: `artifacts/options/`, `artifacts/subfunds_overlay_nav.parquet`,
`artifacts/reports/options_monthly_backtest.md`, `artifacts/reports/options_validation.md`.
Metoda i ograniczenia: [`docs/options_monthly.md`](docs/options_monthly.md).

Wnioski z przebiegu `3ababedef5d8`:

- **Polityka P3 to niemal pełne zabezpieczenie.** Przy λ = 2 nakładka jest collarem w 89–100% miesięcy,
  ze sprzedanym callem ATM o h = 1 dla 78% par (ticker, data). SFk_OV traci ekspozycję na aktywo bazowe: SF1_OV
  ma zwrot −0,05% i zmienność 1,2% rocznie przy koszcie κ ok. 3,7% rocznie, czyli mniej niż WIBOR 1M.
  SFk_OV w USD zachowują głównie ekspozycję na kurs USD/PLN.
- **Wycena zawyża premie sprzedawanych opcji.** Przy `atm_adj = 1` replika BXM wyprzedza indeks Cboe
  o 5,4 p.p. rocznie, a przy `atm_adj` = 0,775 (estymata z kwotowań SPY) o −0,7 p.p. Backtest używa `atm_adj = 1`.
- **Przewaga wariantów `_ov` nie jest odporna.** Kolumny SFk_OV mają średnio 2–8% wagi w strategiach CE,
  a różnica CE skupia się w 2020-09 i 2021-01 (m_t ≈ 2,5). `min_es_ov` trzyma ok. 50% w SF1_OV i ma CE ujemny.
- **Zmiana po przebiegu:** m_t ma górne ograniczenie `implied_vol.m_cap` w `config/options.yaml`: kwantyl 95%
  historii m_s z okna estymacji marginesu S&P 500 (dane ≤ t − 1). Na datach backtestu wiąże w 19 z 226 dat,
  a w 2020-09 i 2021-01 obniża m_t z ok. 2,5 do 1,54 i 1,67.
- **Pozostałe zalecenia przed użyciem wyników `_ov`:** `atm_adj` ≈ 0,8 i β ≈ −0,5 oraz niższe λ (0,1–0,25),
  przy którym decyzje są zróżnicowane.

