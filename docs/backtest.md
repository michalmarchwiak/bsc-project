# Backtest: profile inwestora, modele zależności, MC dla ES i ranking

Metodologia backtestu strategii opartych na modelach zależności (pakiet `pf`, pipeline'y w
`pipelines/`). Wyniki: `artifacts/backtest/<run_id>/` (tabele zbiorcze), raporty w
`artifacts/reports/` (`backtest_summary.md`, `ranking_calibration.md`, `ranking_holdout.md`,
`dependence_models.md`, `es_mc.md`), wykresy w `figures/backtest/`.

## 1. Dane i próby

- Dzienne log-stopy w PLN 2005-01-04 – 2026-10-29 (`data_import.py`, `data_manipulation.py`), 79
  tickerów. AVB (AvalonBay) jest niedostępny: spółkę wycofano z obrotu w 2026 r., Yahoo i Nasdaq
  nie udostępniają już historii, a Stooq (`avb.us`) odrzuca połączenia z kontenera sesji. SF9 (REIT)
  ma więc 9 składników. Kontrola kompletności: `pf.data.quality`.
- SF10: 8 rodzin obligacji oszczędnościowych Skarbu Państwa (tickery `*.SP`). Ich dzienne log-stopy pochodzą
  z indeksów wartości rolowanych pozycji, zbudowanych z warunków emisji (zbiór obrotomierz.pl, dane MF
  i listy emisyjne; `pf.data.savings_bonds`, opis: `docs/savings_bonds.md`). Aktywa SF10 wymagają
  21 sesji w oknie (`min_obs_deterministic`) i notowania w sesji t−1.
- Próby (`config/backtest.yaml`): kalibracyjna 2008-01-01 – 2020-12-31 (156 dat rebalansowania),
  hold-out 2021-01-01 – 2026-10-29 (70 dat; ostatnia, 2026-10-01, to bieżący skład bez okresu realizacji,
  więc ocena obejmuje 69 miesięcy). Data należy do próby, w której leży.
- Daty rebalansowania: pierwsza sesja miesiąca, przed którą jest ≥ 750 sesji (`min_obs`). Przy
  `require_all_subfunds: false` sub-fundusz bez aktywa z 750 obserwacjami w oknie ma wagę 0 do czasu
  dostępności danych (SF1 wchodzi w 2008-02, SF7 – high yield – w 2010-05). Przy ścisłym warunku
  `covers_all_subfunds` pierwszą datą byłby 2010-05-03.

### Uniwersum punktowo w czasie (`pf.universe`, `config/subfunds.yaml`)

Pierwotna lista spółek (uniwersum `fixed`) została dobrana z perspektywy 2026 r., co wprowadza do backtestu
obciążenie doborem ex post. W uniwersum `pit` (domyślnym) sub-fundusze akcyjne SF1, SF2, SF9 i spółki wydobywcze
w SF8 są pulami kandydatów (ok. 40, 120, 36 i 18 spółek), a skład w dacie t wyznacza reguła na danych sprzed t:

- ranking: średnia z 12 ostatnich miesięcznych median dziennej wartości obrotu Close·Volume (waluta notowania),
  z miesięcy zakończonych przed datą rewizji r(t) – pierwszą sesją stycznia;
- skład: `top` spółek o najwyższym rankingu spośród dopuszczonych w oknie estymacji przed t (≥ 750 sesji);
  w SF8 ETF-y na złoto (`core`) należą do składu zawsze, a z rankingu pochodzi 5 spółek;
- skład zmienia się więc raz w roku albo gdy spółka po debiucie uzyska wymaganą historię; sprzedaż spółek
  usuniętych ze składu obciąża koszt transakcyjny.

Marginesy i kopuły daty t obejmują tylko aktywa składu (wymiar jak w uniwersum `fixed`, ok. 80). Skład w każdej
dacie zapisuje `artifacts/backtest/<run_id>/universe.parquet`. Wybór uniwersum: klucz `universe` albo zmienna
`PF_UNIVERSE` (`PF_UNIVERSE=fixed python -m pipelines.backtest …`); zmienia `run_id`.

Ograniczenia reguły: wartość obrotu przybliża wielkość spółki, ale faworyzuje spółki o wysokiej rotacji; pule
zawierają tylko spółki notowane do dziś (spółek wycofanych z obrotu nie ma w danych Yahoo), więc obciążenie
przeżywalnością jest ograniczone, a nie usunięte; listy pul zestawiono ręcznie i należy je zweryfikować
z oficjalnymi składami indeksów.

### Benchmarki rynkowe

`market_benchmarks` w `config/backtest.yaml`: `idx_spx` (ETF SPY – S&P 500 z dywidendami, w PLN) i `idx_wig20`
(indeks WIG20TR ze Stooq; kopia lokalna `data/raw/stooq/wig20tr.csv`). Szeregi `-|idx_*|ALL|base` powstają
w konsolidacji (kupno i trzymanie, koszt zakupu początkowego), wchodzą do rankingu i MCS każdego profilu, nie mają
prognoz ex ante ani testów VaR/ES. WIG20TR jest indeksem (bez kosztów replikacji). Benchmark bez danych jest
pomijany z ostrzeżeniem.

## 2. Opóźnienie wykonania i stopa wolna od ryzyka

Decyzja w sesji t korzysta z danych ≤ t−1. Okres realizacji obejmuje sesje [i(t) + L, i(t_next) + L):
L = 1 (domyślnie) – transakcja na zamknięciu t, okres (t, t_next]; L = 0 (wrażliwość) – [t, t_next).
Scenariusze obejmują H + L dni od końca okna, a stopa okresu – dni L+1, …, L+H (H = 21).

r_f: WIBOR 1M (Stooq `plopln1m`; niedostępny z kontenera), zapasowo średnia miesięczna WIBOR 1M
(Eurostat `irt_st_m`, PL, IRT_M1) przypisana dniom miesiąca, a po jej końcu stopa referencyjna NBP
(archiwum XML NBP). Szereg 2005-01-03 – 2026-10-29 nie ma luk: 5673 dni ze średniej miesięcznej,
21 dni (10.2026) ze stopy referencyjnej. Stopa okresu (ACT/365):

  r_f(t0, t1) = WIBOR1M(t0) · (t1 − t0)[dni] / 365,

t0, t1 – zamknięcia sesji otwarcia i zamknięcia pozycji.

## 3. Modele zależności (krok 1)

Wspólne marginesy AR(1)–GJR-GARCH(1,1) ze skośnym t, estymowane raz na datę (pamięć podręczna),
panel reszt od debiutu najmłodszego aktywa okna (bez wypełniania okresów przed notowaniem; pojedyncze
braki → ẑ = 0), pseudo-obserwacje z rang. Modele (`pf.models.dependence`): `rvine`, `cvine`, `dvine`
(sekwencyjna MNW kopuł parowych, obcięcie na 3 drzewach, rodziny Gauss/t/Clayton/Gumbel z obrotami
i Frank), `gauss`, `student_t`, `indep`, `empirical`. Szczegóły i wzory: `docs/dependence_model.md`.
Panel do estymacji kopuł to domyślnie sumy reszt z k = 5 kolejnych sesji (`dependence.aggregation`; k = 1 –
panel dzienny wcześniejszych przebiegów), co ogranicza wpływ asynchroniczności zamknięć GPW/NYSE i fixingu NBP na
zależność w horyzoncie miesiąca (opis i konsekwencje dla AIC/BIC: `docs/dependence_model.md`; dobór k:
`pipelines.dependence_diagnostics`).
Ocena: AIC/BIC i test Vuonga (R-vine vs pozostałe) na oknie; log-score poza próbą
LS = Σ_t log ĉ(u_t) z PIT filtru GARCH naprzód (`MarginFit.pit_forward`) dla sesji [t, t_next).
Obligacje SF10 mają margines deterministyczny (r_h = μ̂, średnia z 21 ostatnich sesji, σ = 0) i nie wchodzą
do panelu reszt, modeli zależności ani ocen modeli.

## 4. Scenariusze ex ante (krok 3, część ex ante)

Partie po B = 2500 scenariuszy, stop przy max_k SE(ÊS_k)/|ÊS_k| ≤ ε lub N ≥ 50 000 (ε = 0,02;
ostatnia data 0,01); portfele kontrolne: EW w każdym SF i EW między SF:

  SE(ÊS_α) = √( [Var(L | L > VaR_α) + α (ES_α − VaR_α)²] / ((1 − α) N) ).

Wspólne liczby losowe: W ~ U(0, 1) ze strumienia (ziarno, data) wspólnego dla modeli; losowość
specyficzna dla modelu – ziarno (ziarno, data, CRC32(model)). Wszystkie strategie, profile i warianty
danej daty korzystają z tych samych scenariuszy danego modelu. Kopuła generuje wartości tylko dla aktywów
stochastycznych; stopa aktywa SF10 jest stała we wszystkich scenariuszach, R_i = exp(H·μ̂_i) − 1.

## 5. Profile, ekwiwalent pewności i kara za zależność (krok 2)

| id | nazwa | γ | κ | c_SF | c_LT | W_max |
|---|---|---:|---:|---:|---:|---:|
| P1 | konserwatywny | 10 | 1.00 | 0.20 | 0.05 | 0.40 |
| P2 | umiarkowany | 7 | 0.60 | 0.25 | 0.07 | 0.45 |
| P3 | zrównoważony | 5 | 0.35 | 0.30 | 0.10 | 0.50 |
| P4 | dynamiczny | 3 | 0.15 | 0.40 | 0.15 | 0.60 |
| P5 | agresywny | 1.5 | 0.05 | 0.50 | 0.25 | 0.75 |

Ekwiwalent pewności CRRA na scenariuszach miesięcznych stóp prostych R_s:

  CE_γ(w) = ( (1/S) Σ_s (1 + R_s w)^{1−γ} )^{1/(1−γ)} − 1,   CE_1(w) = exp((1/S) Σ_s ln(1 + R_s w)) − 1,

∇CE_γ(w) = M^{γ/(1−γ)} (1/S) Σ_s (1 + R_s w)^{−γ} R_s, M = (1/S) Σ_s (1 + R_s w)^{1−γ}.
Miara zależności D(w) = wᵀMw, M z pseudo-obserwacji scenariuszy V_si = rank(R_si)/(S + 1):
Λ_ij(q) = (1/(Sq)) Σ_s 1{V_si ≤ q}·1{V_sj ≤ q} (q = 0,05; domyślnie) lub ρ_S = corr(V).
Obie macierze są PSD, CE_γ jest wklęsła, więc problem jest wklęsły.

Poziom 1 (SF k): max_w CE_γ(w) − κ σ_k^EW D_k(w), Σw = 1, 0 ≤ w_i ≤ max(c_SF, 1/n_k);
σ_k^EW – odchylenie standardowe stopy SF przy równych wagach.
Poziom 2: R^SF_{s,k} = R^{(k)}_s w_k, max_W CE_γ(W) − κ σ^EW_SF D^SF(W), ΣW = 1, 0 ≤ W_k ≤ W_max,
W_k · max_i w_{k,i} ≤ c_LT. Ograniczenie look-through jest ograniczeniem górnym W_k ≤ min(W_max, c_LT/m_k);
przy Σ_k u_k < 1 c_LT jest luzowany proporcjonalnie (minimalny czynnik s ≥ 1), z flagą `lt_relaxed`.
Rozwiązanie: SLSQP z gradientem analitycznym, start z EW (lub u/Σu).

Strategie (każdy model × profil): `ce_dep` (proponowana), `ce` (κ = 0), `maxret_es` (max E[R] przy
ES_α ≤ ES(EW), LP Rockafellara–Uryaseva), `min_es` (min ES_α, LP); benchmarki bez kopuły: `ew`
(1/N w SF i między SF, bez limitów) i `mv_hist` (max μᵀw − (γ/2) wᵀΣw, μ, Σ = 21·średnia/kowariancja
dziennych stóp okna, Σ z shrinkage Ledoita–Wolfa do m·I, oba poziomy, limity profilu). Siatka:
7 × 4 × 5 + 1 + 5 = 146 szeregów. Przy włączonej nakładce opcyjnej (`options_overlay` w
`config/backtest.yaml`) dochodzą warianty `<strategia>_ov` dla P3–P5: 7 × 4 × 3 = 84, razem 230 szeregów
(`docs/options_monthly.md`). Prognozy ex ante benchmarków – ze scenariuszy R-vine.

**Założenie.** Dekompozycja dwupoziomowa optymalizuje CE pojedynczego SF z γ inwestora. To heurystyka,
a nie optimum łączne względem wag look-through.

## 6. Backtest (krok 4)

Dla każdej daty: okno 1260 sesji ≤ t−1 → marginesy → 7 modeli → scenariusze → optymalizacja → prognozy
ex ante (VaR, ES, kwantyle i ogon – najgorsze ⌈(1−α)N⌉ strat, float32). Zapis po dacie
(`dates/<t>/`, wznawialne), równoległość po datach (`--jobs`, GARCH jobs = 1, BLAS 1 wątek).

Wagi tickerów (poziom 1) i sub-funduszy (poziom 2) są wyznaczane od nowa w każdej dacie rebalansowania
regułą danej strategii (dla `ce`/`ce_dep` – maksymalizacja CE_γ profilu, z karą κ dla `ce_dep`).
Wagi w czasie: `artifacts/backtest/<run_id>/weights_tickers.parquet` (wszystkie szeregi bazowe) i
`artifacts/reports/weights/` (CSV dla strategii głównej, opis w `README.md`).

Realizacja: kupno i trzymanie z dryfem wag look-through x = Σ_k W_k w_k; V_h = Σ_i x_i Π_{j≤h}(1 + r_ij);
koszt c·Σ_i|x_i − x̃_i| + Σ_i f_i(t)·(x̃_i − x_i)⁺ (c = 0,1%; zakup początkowy c; f_i – opłata za wcześniejszy
wykup obligacji SF10 jako ułamek wartości, dla pozostałych aktywów 0), x̃ – wagi po dryfie; stopy dzienne
i miesięczne brutto/netto, analogicznie dla każdego SF. Tabela `periods.parquet` zawiera kolumnę
`redemption_cost`.

PIT zrealizowanej straty u_t = F̂_t(L_t) (dokładnie z ogona, poniżej VaR z siatki kwantyli). Testy
(osobno kalibracja i hold-out, `pf.backtest.risk_tests`): Kupiec (LR i dokładny dwumianowy),
Christoffersen (niezależność, pokrycie warunkowe), Acerbi–Szekely
Z₂ = 1 − (1/(T(1−α))) Σ_t L_t 1{L_t > VaR_t}/ES_t (wartości krytyczne z symulacji pod H0 z zapisanych
ogonów prognoz), Du–Escanciano H_t = (u_t − α)1{u_t ≥ α}/(1 − α), test bezwarunkowy
(E[H] = (1−α)/2, Var[H] = (1−α)(1/3 − (1−α)/4)) i warunkowy (Box–Pierce, opóźnienia 1 i 5).

Wrażliwość (tylko daty kalibracji, te same scenariusze): L = 0 (te same ścieżki dzienne przesunięte o
dzień; strategie `ce_dep`, `ce` i benchmarki), q = 0,10, ρ_S, siatka κ ∈ {0, 0,05, 0,15, 0,35, 0,6, 1, 2}
(`ce_dep`); wybór κ z ograniczeniem κ_P1 ≥ … ≥ κ_P5 (programowanie dynamiczne) jest raportowany
informacyjnie.

**Zamrożenie.** Parametry meta (γ, κ, limity, q, ε, obcięcie vine'a) są wartościami wyjściowymi z
`config/profiles.yaml` i `config/models.yaml`, ustalonymi przed backtestem i niedostrajanymi na
kalibracji. Przed pierwszą datą hold-out zapisywany jest `frozen_params.json` (skrót konfiguracji);
przebieg hold-out z inną konfiguracją jest odrzucany. Modele (GARCH, kopuły) są nadal estymowane
co miesiąc w oknie kroczącym – to część strategii.

## 7. MC ex post (krok 3, część ex post)

`pipelines.expost_mc`: dla każdego szeregu (portfele bazowe i ich SF) i próby, na dziennych stopach netto:
FHS (GJR-GARCH(1,1) skew-t, bootstrap reszt, stan startowy z losowej daty) i bootstrap stacjonarny
Politisa–Romano (`arch`, długość bloku max(b̂(r), b̂(r²)) z `optimal_block_length`); dla strategii głównej
(rvine × ce_dep) także wielowymiarowe MC (marginesy GARCH + R-vine na szeregach SF, SF10 ze stałą stopą; wagi W z ostatniej
daty i średnie). N adaptacyjnie (SE(ÊS)/|ÊS| ≤ 0,01, N ≤ 200 000). Wynik: miesięczne VaR i ES 97,5% z SE,
porównanie z ES ex ante (średnia prognoz) i historycznym. Szereg strategii o zmiennych wagach nie jest
ściśle stacjonarny.

## 8. Ranking (krok 5)

Miesięczne stopy netto w PLN, osobno kalibracja i hold-out; wnioski na hold-out. Miara główna: CE
zrealizowany per profil, 1 + r̃_t = (1 + r_t)/(1 + r_f,t), CE = ((1/T) Σ (1 + r̃_t)^{1−γ})^{1/(1−γ)} − 1,
annualizacja (1 + CE)^{12} − 1. Dodatkowo: Sharpe (×√12, SE Lo 2002), Sortino, Calmar, max DD, zwrot
i zmienność roczne, Omega(0), STARR = E[r − r_f]/ES ex post (FHS), obrót, koszty, p-wartości testów.
Wnioskowanie: MCS (`arch.bootstrap.MCS`, α = 0,10, bootstrap stacjonarny) per profil na stracie
ℓ_t = −u_γ(1 + r̃_t); test różnicy Sharpe'a względem `ew` (Ledoit–Wolf 2008, studentyzowany bootstrap
blokowy cykliczny, blok 3 mies.). Ranga złożona – średnia ważona rang (wagi w `config/backtest.yaml`).
Ranking modeli zależności: średnie AIC/BIC, wygrane w teście Vuonga, MCS na dziennej stracie −log-score,
odsetek zaliczonych testów VaR/ES, średnia ranga CE portfeli.

## 9. Wyniki (przebieg `83254f06e339`)

**Uwaga.** Wyniki poniżej pochodzą z przebiegu sprzed wprowadzenia uniwersum `pit`, agregacji panelu reszt
i benchmarków rynkowych: uniwersum `fixed`, panel dzienny (k = 1). Nie odpowiadają bieżącej konfiguracji.

226 dat (kalibracja 156, hold-out 70; ocena 156 + 69 miesięcy), 230 szeregów bazowych (146 bez nakładki
i 84 warianty `_ov` dla P3–P5) + 391 wariantów wrażliwości, 52,6 h CPU. Mediana N = 12 500 scenariuszy
(ostatnia data 35–50 tys.). Luzowanie limitów c_LT/W_max nie było potrzebne.

Średni zrealizowany CE [% rocznie] po 7 modelach (hold-out; kalibracja w nawiasie):

| profil | ce_dep | ce | maxret_es | min_es | ce_dep_ov | ce_ov | ew | mv_hist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 4,53 (5,54) | 3,86 (5,04) | 3,35 (3,47) | 0,36 (1,98) | – | – | 3,52 (3,65) | 2,37 (5,12) |
| P2 | 5,48 (8,06) | 6,25 (7,76) | 4,75 (6,21) | 0,74 (3,40) | – | – | 4,37 (5,23) | 3,52 (6,69) |
| P3 | 7,62 (10,99) | 7,21 (11,39) | 5,47 (7,82) | 1,22 (3,72) | 8,14 (12,13) | 7,33 (12,48) | 4,93 (6,30) | 4,89 (10,89) |
| P4 | 10,25 (16,13) | 10,93 (16,02) | 7,02 (9,22) | 1,73 (3,88) | 10,69 (17,21) | 11,54 (17,08) | 5,50 (7,39) | 8,21 (15,05) |
| P5 | 16,10 (22,19) | 16,19 (21,95) | 8,41 (10,61) | 2,20 (3,64) | 17,31 (23,70) | 17,49 (23,33) | 5,92 (8,23) | 12,11 (17,56) |

Strategie CE dominują pozostałe reguły w hold-out. W kalibracji `mv_hist` wyprzedza `ce` w P1. Kara za
zależność podnosi CE w kalibracji (P1, P2, P4, P5), a w hold-out tylko w P1 i P3. Nakładka opcyjna
podnosi średni CE strategii `ce`, `ce_dep` i `maxret_es` w P3–P5 w obu próbach, a obniża CE `min_es`.
MCS w hold-out obejmuje wszystkie 30 szeregów w P1–P2 oraz 34–55 z 58 szeregów w P3–P5. Test
Ledoita–Wolfa nie odrzuca równości Sharpe'a z `ew` dla strategii głównej. Przy 69 miesiącach różnice
między strategiami CE są statystycznie nierozróżnialne. Szczegóły: `artifacts/reports/`.

## 10. Uruchomienie

```bash
python -m pipelines.backtest --sample calibration --jobs 4   # daty kalibracji (wznawialne)
python -m pipelines.backtest --sample holdout --jobs 4       # zamrożenie parametrów, daty hold-out
python -m pipelines.backtest --consolidate-only               # tabele zbiorcze, PIT, testy VaR/ES
python -m pipelines.expost_mc --jobs 4                        # MC ex post → es_mc.md
python -m pipelines.ranking                                   # ranking, raporty, wykresy
```

## 11. Ograniczenia

- **Asynchroniczność.** GPW zamyka się o 17:00, NYSE o 22:00, a fixing NBP jest ok. 11:50. Dzienna zależność
  jest więc zniekształcona (między rynkami zaniżona, między aktywami w USD przeliczonymi na PLN zawyżona),
  a przeliczenie na PLN używa kursu z innej chwili niż cena. Agregacja panelu reszt (k = 5) ogranicza ten efekt.
- **Uniwersum.** Reguła obrotowa w uniwersum `pit` korzysta tylko z danych sprzed daty decyzji, ale pule obejmują
  wyłącznie spółki notowane do dziś; w uniwersum `fixed` lista spółek jest dobrana ex post.
- **Heurystyka dwupoziomowa.** CE pojedynczego SF z γ inwestora nie jest CE portfela.
- **Niska moc testów.** Przy próbie miesięcznej oczekiwana liczba przekroczeń VaR 97,5% jest mała:
  3,9 w kalibracji (156 mies.) i 1,7 w hold-out (69 mies.).
- **Krótki hold-out.** Obejmuje ok. 69 miesięcy, więc różnice CE i Sharpe'a są obarczone dużym błędem
  (zob. MCS i test Ledoita–Wolfa).
- **Oczekiwane stopy.** Wynikają ze stałych AR(1) w oknie (duży błąd estymacji). W okresach wysokiej
  zmienności E[R] scenariuszy rośnie przez wypukłość exp(·).
- **Stopa wolna od ryzyka.** Dzienny WIBOR 1M był niedostępny, więc r_f to średnia miesięczna
  (Eurostat), a w 10.2026 stopa referencyjna NBP.
- **Brak AVB** w SF9.
- **Obligacje SF10.** Indeks odzwierciedla rolowaną pozycję jednego rocznika, a nie serię bieżącą. Prognoza
  stopy z 21 sesji pomija zmianę stopy w nowym okresie odsetkowym. W miesiącu wygaśnięcia indeksu
  (DOS 2023-01, TOZ 2024-05) środki po wykupie nie są oprocentowane do następnej daty rebalansowania.
  Szczegóły: `docs/savings_bonds.md`.
- **Stały horyzont.** Horyzont scenariuszy H = 21 sesji jest stały, a okres realizacji ma 19–23 sesje.
- **Zależność ścieżek w bootstrapie.** W bootstrapie stacjonarnym sąsiednie ścieżki 21-dniowe z jednej
  replikacji są słabo zależne, więc SE jest lekko zaniżony.
