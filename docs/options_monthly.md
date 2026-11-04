# Miesięczna nakładka opcyjna na tickery sub-funduszy

Moduł wycenia w każdej dacie rebalansowania hipotetyczne opcje europejskie bliskie ATM na każdy ticker
sub-funduszy SF1–SF9, rozlicza je w kolejnej dacie i włącza do NAV sub-funduszu zgodnie z polityką
decyzyjną (brak / put / covered call / collar). Warianty SFk_OV wchodzą jako dodatkowe kolumny do
optymalizacji składu funduszu.

**Opcje są hipotetyczne.** Dla większości tickerów (spółki GPW, ETF-y obligacyjne i regionalne, spot
surowców) nie istnieją notowane opcje miesięczne. Wycena ma charakter modelowy, a σ_imp jest
odtwarzana, a nie obserwowana.

## Uruchomienie

```bash
python -m pipelines.options_monthly --jobs 4               # wszystkie etapy, ok. 30–40 min na 4 rdzeniach
python -m pipelines.options_monthly --refresh-data         # najpierw odświeżenie danych z sieci (Cboe, Yahoo, FRED)
python -m pipelines.options_monthly --stages consolidate report   # wybrane etapy (wyniki dat z pamięci)
python -m pipelines.options_monthly --dates 2008-10-01 --stages ledger --force
```

| etap | zawartość |
|---|---|
| `ledger` | krok każdej daty (równolegle, wznawialnie, skrót danych ≤ t_{j+1} i konfiguracji w `done.json`) |
| `consolidate` | rejestr, sygnały i pozycje P0–P2, decyzje, NAV SF |
| `portfolio` | poziom 1–2 bez i z kolumnami SFk_OV, realizacja stóp portfeli |
| `validation` | replikacja Cboe BXM i PPUT, estymata `atm_adj` z kwotowań SPY |
| `annual` | scenariusze roczne SF z nakładką miesiąc po miesiącu (ostatnia data lub `--annual-date`) |
| `report` | `artifacts/reports/options_monthly_backtest.md` |

Parametry: `config/options.yaml`, sekcja `monthly_overlay`. Testy: `tests/test_options_overlay_*.py`.

## Założenia

1. **Kalendarz.** Daty t_j są datami rebalansowania backtestu (pierwsza sesja miesiąca w sumie kalendarzy GPW
   i NYSE). Przy opóźnieniu wykonania L = 1 opcja jest kupowana lub sprzedawana na zamknięciu t_j i wygasa na
   zamknięciu t_{j+1}, czyli w tych samych okresach realizacji co sub-fundusze w backteście
   (`pf.backtest.data.holding_bounds`). Czas do wygaśnięcia τ_j = n_j/252, gdzie n_j to liczba sesji
   w (t_j, t_{j+1}]. Modele są estymowane na danych ≤ t_j − 1, a ceny, indeksy zmienności, stopy i kursy
   pochodzą z zamknięcia t_j.
2. **Typ opcji.** Wyłącznie opcje europejskie z rozliczeniem pieniężnym po cenie zamknięcia t_{j+1}. Nie ma
   wcześniejszego wykonania ani przeszacowania w trakcie miesiąca. Po rozliczeniu nakładka jest ustanawiana od nowa.
3. **Opcje bliskie ATM.** Siatka K/S_t ∈ {0,95; 0,975; 1,00; 1,025; 1,05} z warunkiem |ln(K/F)| ≤ 0,06.
   Przy pierwotnym progu 0,05 put 0,95 byłby zawsze odrzucony, bo ln(0,95·S/F) = −0,0513 − (r − q)τ.
   Wiersze spoza pasma zostają w rejestrze z `in_band = False`. Nakładka używa putów K/S ≤ 1 i calli K/S ≥ 1.
4. **Covered only.** h^P, h^C ∈ [0, 1], a sumy po strike'ach są ≤ 1 dla każdego tickera. Nie ma nagich
   opcji ani dźwigni.
5. **Zakres.** Nakładką objęte są SF1–SF9. SF10 (obligacje oszczędnościowe) jest wyłączony, bo te
   instrumenty nie są notowane, a ich stopa jest deterministyczna.

## Dane

| wielkość | źródło |
|---|---|
| ceny S (waluta notowania) | `data/processed/log_returns_local.csv`: S = A·exp(Σ r), ceny skorygowane (TR), więc q = 0 |
| USD/PLN | `data/raw/fx_nbp.csv` (kurs średni NBP z ostatniej sesji tickera) |
| VIX, VIX9D, GVZ, OVX | `artifacts/market/vol_indices.parquet` (Cboe, zapasowo Yahoo) |
| stopa PLN | WIBOR 1M (`artifacts/market/riskfree_pln.parquet`, ACT/365) |
| stopa USD | ^IRX, zapasowo FRED DTB3 (`pf.data.usd_rate`, ACT/360) |
| S&P 500, S&P 500 TR | `artifacts/market/underlyings.parquet` (m_t, walidacja) |
| BXM, PPUT | `artifacts/market/cboe_strategy_indices.parquet` (walidacja; brak danych → `status: skipped`) |

Stopa ciągła na okres opcji: r = ln(1 + y·d/basis)/τ, gdzie d to liczba dni kalendarzowych okresu. Czynnik
dyskontowy e^{−rτ} równa się wtedy 1/(1 + y·d/basis). Dla tickera w USD stopa w PLN spełnia
G = g·f (g – brutto-stopa w USD, f – zmiana kursu). Ta tożsamość obowiązuje dokładnie dla
`log_returns_pln.csv`.

## Wycena (`pf.options.implied_vol`, `pf.options.ledger`)

BSM z ciągłą stopą dywidendy (`pf.options.bs`), q = 0:

    F = S e^{(r−q)τ},  d₁ = [ln(S/K) + (r − q + σ²/2)τ]/(σ√τ),  d₂ = d₁ − σ√τ,
    C = S e^{−qτ} N(d₁) − K e^{−rτ} N(d₂),  P = K e^{−rτ} N(−d₂) − S e^{−qτ} N(−d₁).

**Prognoza GARCH.** Średnia zmienność na okres opcji (AR(1)–GJR-GARCH(1,1) skew-t, okno ≤ t_j − 1):

    σ̂_{i,t}(τ) = √[(252/n) Σ_{h=L+1}^{L+n} E_t σ²_{i,t+h}],   E_t σ²_{t+h} = ω + ρ E_t σ²_{t+h−1},
    ρ = α + β + γ κ₋,   κ₋ = E[z² 1{z < 0}].

Wielkość κ₋ jest liczona kwadraturą z gęstości skośnego t Hansena, a dla λ = 0 wynosi κ₋ = 1/2.

**Mnożnik premii za ryzyko zmienności:** m_t = (VIX_t/100)/σ̂_{SPX,t}(21), z marginesem S&P 500 z okna ≤ t_j − 1.

**Źródło σ_imp** (`iv_source`):
* GVZ dla GLD, IAU i SGOL: σ = atm_adj·GVZ/100. Przed 09.2009 GVZ nie istnieje i stosowane jest `proxy`.
* VIX dla S&P 500: wariancja całkowita jest interpolowana między VIX9D i VIX. Źródło służy tylko do m_t
  i walidacji, bo SF nie zawierają SPY. OVX (ropa) nie jest używany, bo w SF nie ma instrumentu na ropę.
* `proxy` dla pozostałych tickerów: σ_imp = clip(m_t·σ̂_{i,t}(τ), σ_min, σ_max).
* Opcjonalna korekta skośności: σ_imp(K) = σ_imp + β ln(K/F). Domyślnie β = 0, a β = −0,5 służy do analizy wrażliwości.

**Koszt** κ = 3% premii: kupno kosztuje P(1 + κ), sprzedaż daje C(1 − κ). Opcje na tickery w USD są wyceniane
w USD, premia jest przeliczana po kursie z t_j, a wypłata po kursie z t_{j+1}.

**Rejestr** (`artifacts/options/ledger.parquet`, cała siatka niezależnie od polityki): `date, t, t_next,
subfund, ticker, currency, type, moneyness, K, S_t, F, tau, r, q, sigma_imp, sigma_atm, sigma_hat, iv_source,
log_kf, in_band, premium, premium_rel, delta, gamma, vega, S_next, payoff, payoff_rel, fx_t, fx_next, pnl_pln,
pnl_rel_pln`. Wielkość `pnl_pln` to P&L długiej pozycji przed kosztem κ.

## NAV sub-funduszu (`pf.options.nav`)

Wagi wewnętrzne w_i są równe między tickerami SF dostępnymi w t_j, zgodnie z regułą macierzy A. Oznaczenia:
p = P/S, c = C/S.

    D_i = 1 + Σ_k h^P_{ik} p_{ik} (1 + κ) − Σ_k h^C_{ik} c_{ik} (1 − κ),   V′_j = V_j / Σ_i w_i D_i,
    n_i = w_i V′_j / S_{i,t_j},
    V_{j+1} = Σ_i n_i [S_{i,t+1} + Σ h^P (K^P − S_{i,t+1})⁺ − Σ h^C (S_{i,t+1} − K^C)⁺]   (PLN).

Ekspozycja skorygowana o deltę wynosi e_i = w′_i(1 + Σ h^P Δ^P − Σ h^C Δ^C), gdzie w′_i = w_i/Σ w D.
Testy kontrolne sprawdzają, że przy h ≡ 0 NAV równa się NAV bazowego SF oraz że portfel jest
samofinansujący (akcje + opcje + koszty = V_j).

## Polityki (`pf.options.policy`)

Wszystkie sygnały pochodzą z danych ≤ t_j. Percentyle są liczone w oknie rozszerzającym się dat t_j
i wymagają co najmniej 36 wcześniejszych miesięcy; wcześniej P2 = brak. Pierwsza data backtestu to 2008-01,
więc P2 działa od 2011-01.

* **P0:** brak nakładki (benchmark).
* **P1:** stały put K = 0,975·S z h̄ = 0,5. Wariant `P1_cc` to stały covered call K = 1,025·S z h̄ = 0,5.
* **P2:** sygnały Q_t (percentyl m_t), T_{i,t} = S/SMA_10M − 1 i V_{i,t} (percentyl σ̂). Reguły:
  * put przy Q ≤ 0,3 (K/S = 1,00) lub przy T < 0 i Q ≤ 0,7 (K/S = 0,975);
  * collar (put 0,975 + call 1,05) przy T < 0 i Q > 0,7;
  * covered call 1,025 przy Q ≥ 0,7, T > 0 i V ≤ 0,8.

  Progi są ustalone ex ante, bez dostrajania. Wybór strike'a w regule put to doprecyzowanie specyfikacji
  (`rules.put_cheap`, `rules.put_trend`).
* **P3:** LP liczone dla każdego SF i daty t_j na S = 5000 scenariuszach (model poniżej):

      R⁽ˢ⁾ = Σ_i w_i [G_i⁽ˢ⁾ − 1] + Σ_{j∈P} h_j w_i [f_i⁽ˢ⁾(m_j − g_i⁽ˢ⁾)⁺ − p_j(1 + κ)]
                                 − Σ_{j∈C} h_j w_i [f_i⁽ˢ⁾(g_i⁽ˢ⁾ − m_j)⁺ − c_j(1 − κ)],
      max E[R] − λ CVaR_{97,5%}(R)  (R–U),  Σ_{j∈P(i)} h_j ≤ 1,  Σ_{j∈C(i)} h_j ≤ 1.

  Liniowość w h to przybliżenie pierwszego rzędu względem mianownika Σ w_i D_i. Pominięty składnik jest
  rzędu premia × stopa. Pasmo bezczynności: nakładka jest ustanawiana tylko przy poprawie funkcji celu > ε = 2·10⁻⁴.
  λ = 2, wrażliwość obejmuje λ ∈ {0,1; 0,25; 1; 3}.

**Model scenariuszy P3** (`pf.options.scenarios`) składa się z marginesów GJR-GARCH w walucie notowania,
marginesu USD/PLN i kopuły t na resztach. Dryf jest konserwatywny: E[g_i] = exp((r_i + SR·σ̂_i)τ) przy SR = 0,3
oraz E[f] = exp((r_PLN − r_USD)τ). Skalowanie stałą zachowuje rangi, więc decyzję napędza premia za ryzyko
zmienności (σ_imp vs σ w mierze P) i ogon rozkładu, a nie prognoza kierunku.

**Skala λ.** W horyzoncie miesiąca CVaR_{97,5%} akcji (≈ 10–17%) jest o rząd wielkości większy od E[R]
(≈ 0,5–1%). Przy λ ≥ 1 LP wybiera więc zwykle niemal pełne zabezpieczenie: sprzedaż calla ATM z h^C = 1 i kupno
puta. Taka konwersja (put K + sprzedany call K) jest wyceniana bez wpływu σ (parytet put–call), a jej koszt
to κ(P + C) i utracona premia za ryzyko. Warianty λ = 0,1 i 0,25 pokazują kompromis między zwrotem a ryzykiem.

## Integracja z portfelem (`pf.options.portfolio`)

* **Poziom 1:** skład SF bez zmian (`optimize_two_level`).
* **Poziom 2:** dla profili ≥ P3 (`ov_min_profile`) kolumny SFk_OV są dodatkowymi kolumnami. Ich stopa
  scenariuszowa to dokładny NAV z nakładką przy wagach w_k i pozycjach wybranej polityki
  (`monthly_overlay.selected`, domyślnie P3). Ograniczenie W_k + W_k^OV ≤ u_k łączy limit W_max i limit
  look-through. Kolumna SFk_OV pojawia się tylko przy aktywnej nakładce w t_j, w przeciwnym razie SFk_OV ≡ SFk.
  Decyzja nakładki pochodzi z SF przy wagach równych i nie jest optymalizowana łącznie z w_k (heurystyka).
* **Strategie:** domyślnie `ce_dep` i `maxret_es` (lista `portfolio.strategies`). W `maxret_es` limit ES jest ten
  sam dla wariantu z nakładką i bez niej: ES portfela równych wag bazowych SF.
* **Ten sam budżet ES ex ante** (`r_es_matched`): wariant o wyższym ÊS jest mieszany z lokatą po r_f,
  a = (ES_cel + r_f)/(ES + r_f).
* **Wariant SOCP** z limitem σ nie występuje w repozytorium, więc wiążący jest ES ze scenariuszy.
* **Scenariusze roczne** (`pf.options.paths`): nakładka jest stosowana miesiąc po miesiącu z
  σ_imp⁽ˢ⁾ = m̄·σ̂⁽ˢ⁾ (m̄ to mediana historyczna m_t, σ̂⁽ˢ⁾ pochodzi z rekursji GARCH w ścieżce). Symulacja
  obejmuje polityki P0–P2 i `P3_frozen` (średnie h z backtestu P3); LP nie jest rozwiązywane w ścieżkach.

## Backtest i walidacja (`pf.backtest.overlay`, `pf.options.validation`)

* **Miary SF × polityka × wariant:** zwrot i zmienność roczne, Sharpe i Sortino (r_f = WIBOR 1M okresu),
  max DD, zrealizowany ES 97,5% stóp miesięcznych, suma premii netto i kosztu κ, odsetek miesięcy
  z nakładką i jej typ. Okna stresowe: 2008-09 – 2009-03, 2020-02 – 2020-04 i 2022.
* **Wrażliwość:** σ_imp ± 3 p.p. (tickery proxy), β = −0,5, κ ∈ {1%, 5%} dla P1, P1_cc, P2 i P3. Pozycje P1 i P2
  się nie zmieniają, a P3 jest rozwiązywane ponownie.
* **Replikacja BXM** (sprzedany call ATM) **i PPUT** (put 95%) na S&P 500 tym samym silnikiem: rolowanie
  w 3. piątek, rozliczenie po zamknięciu, q z różnicy indeksu TR i cenowego. Tracking error liczony jest na
  stopach między rolowaniami: rocznie i w oknach stresowych. Raport porównuje też wariant z `atm_adj`
  oszacowanym na kwotowaniach SPY (DoltHub lub zrzuty yfinance) i wariant z β.

## Artefakty

| plik | zawartość |
|---|---|
| `artifacts/options/ledger.parquet` | rejestr opcji (cała siatka, wszystkie daty i tickery) |
| `artifacts/options/overlay_decisions.parquet` | pozycje polityk (wariant bazowy) z w, e, K, premią, typem |
| `artifacts/subfunds_overlay_nav.parquet` | stopy i NAV SF: polityka × wariant (r_base, r_ov, koszty, typ, r_f) |
| `artifacts/options/{signals,market_state,lp_info}.parquet` | sygnały P2, m_t i stan rynku, diagnostyka LP P3 |
| `artifacts/options/portfolio_{weights,exante,periods}.parquet` | wagi poziomu 2, prognozy ex ante, stopy portfeli |
| `artifacts/options/annual_mc.parquet` | rozkład stóp rocznych SF z nakładką (ostatnia data) |
| `artifacts/reports/options_monthly_backtest.md` | raport backtestu, wrażliwości, portfeli, walidacji |
| `artifacts/reports/options_validation.{csv,md}` | tracking error BXM i PPUT |

Parametry (siatka, κ, pasmo, polityki, progi, λ, m̄, atm_adj i jego estymata) oraz wersje pakietów trafiają do
`artifacts/manifest.json` (klucze `options/*`, `subfunds_overlay_nav`, `reports/options_*`). Wyniki per data
(`artifacts/options/dates/`) i pamięć podręczna modeli (`artifacts/options/cache/`) są poza repozytorium.

## Ograniczenia

- **Opcje hipotetyczne.** Na większość tickerów nie ma notowanych opcji miesięcznych, więc wyniki nie
  uwzględniają płynności, spreadów ani dostępności strike'ów.
- **σ_imp odtwarzana modelowo** (GVZ albo proxy GARCH × m_t), a nie obserwowana. Przeniesienie premii za ryzyko
  zmienności z rynku USA (VIX vs GARCH S&P 500) na spółki GPW, ETF-y obligacyjne i regionalne to założenie.
- **VIX ≠ σ_ATM.** VIX przewyższa zmienność ATM, więc przy `atm_adj = 1` premie opcji ATM są zawyżone, co widać
  w replikacji BXM. Estymata `atm_adj` z kwotowań SPY jest zapisywana w manifeście, ale nie jest stosowana
  automatycznie.
- **Skośność.** Ograniczenie do opcji bliskich ATM zmniejsza błąd skośności, ale go nie usuwa. Przy β = 0 puty
  OTM są niedowartościowane.
- **Ryzyko zmienności implikowanej.** Opcja jest trzymana do wygaśnięcia, więc model nie ujmuje ryzyka zmian
  zmienności implikowanej w trakcie miesiąca.
- **Waluta.** Put na ticker w USD chroni cenę w USD, a nie kurs USD/PLN.
- **Daty bez sesji GPW.** Pierwsza sesja miesiąca może być dniem bez sesji GPW. Cena tickera GPW pochodzi wtedy
  z poprzedniej sesji, jak w backteście.
- **Heurystyka P3.** P3 to LP na liniowym przybliżeniu stopy, przy wagach równych w SF. Integracja
  z portfelem stosuje te same h przy wagach w_k strategii.
- **Krótka historia P2.** P2 wymaga 36 miesięcy historii m_t, więc kryzys 2008 jest objęty tylko przez P1 i P3.
