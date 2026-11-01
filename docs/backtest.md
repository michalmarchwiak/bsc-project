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
- Próby (`config/backtest.yaml`): kalibracyjna 2008-01-01 – 2020-12-31 (156 dat rebalansowania),
  hold-out 2021-01-01 – 2026-10-29 (70 dat; ostatnia, 2026-10-01, to bieżący skład bez okresu realizacji,
  więc ocena obejmuje 69 miesięcy). Data należy do próby, w której leży.
- Daty rebalansowania: pierwsza sesja miesiąca, przed którą jest ≥ 750 sesji (`min_obs`). Przy
  `require_all_subfunds: false` sub-fundusz bez aktywa z 750 obserwacjami w oknie ma wagę 0 do czasu
  dostępności danych (SF1 wchodzi w 2008-02, SF7 – high yield – w 2010-05). Przy ścisłym warunku
  `covers_all_subfunds` pierwszą datą byłby 2010-05-03.

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
Ocena: AIC/BIC i test Vuonga (R-vine vs pozostałe) na oknie; log-score poza próbą
LS = Σ_t log ĉ(u_t) z PIT filtru GARCH naprzód (`MarginFit.pit_forward`) dla sesji [t, t_next).

## 4. Scenariusze ex ante (krok 3, część ex ante)

Partie po B = 2500 scenariuszy, stop przy max_k SE(ÊS_k)/|ÊS_k| ≤ ε lub N ≥ 50 000 (ε = 0,02;
ostatnia data 0,01); portfele kontrolne: EW w każdym SF i EW między SF:

  SE(ÊS_α) = √( [Var(L | L > VaR_α) + α (ES_α − VaR_α)²] / ((1 − α) N) ).

Wspólne liczby losowe: W ~ U(0, 1) ze strumienia (ziarno, data) wspólnego dla modeli; losowość
specyficzna dla modelu – ziarno (ziarno, data, CRC32(model)). Wszystkie strategie, profile i warianty
danej daty korzystają z tych samych scenariuszy danego modelu.

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
7 × 4 × 5 + 1 + 5 = 146 szeregów. Prognozy ex ante benchmarków – ze scenariuszy R-vine.

**Założenie.** Dekompozycja dwupoziomowa optymalizuje CE pojedynczego SF z γ inwestora. To heurystyka,
a nie optimum łączne względem wag look-through.

## 6. Backtest (krok 4)

Dla każdej daty: okno 1260 sesji ≤ t−1 → marginesy → 7 modeli → scenariusze → optymalizacja → prognozy
ex ante (VaR, ES, kwantyle i ogon – najgorsze ⌈(1−α)N⌉ strat, float32). Zapis po dacie
(`dates/<t>/`, wznawialne), równoległość po datach (`--jobs`, GARCH jobs = 1, BLAS 1 wątek).

Realizacja: kupno i trzymanie z dryfem wag look-through x = Σ_k W_k w_k; V_h = Σ_i x_i Π_{j≤h}(1 + r_ij);
koszt c·Σ_i|x_i − x̃_i| (c = 0,1%; zakup początkowy c), x̃ – wagi po dryfie; stopy dzienne i miesięczne
brutto/netto, analogicznie dla każdego SF.

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
(rvine × ce_dep) także wielowymiarowe MC (marginesy GARCH + R-vine na 9 szeregach SF, wagi W z ostatniej
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

## 9. Wyniki (przebieg `4010467a06e7`)

226 dat (kalibracja 156, hold-out 70; ocena 156 + 69 miesięcy), 146 szeregów bazowych + 391 wariantów
wrażliwości, 42 h CPU (4 procesy: ok. 11,5 h). Mediana N = 12 500 scenariuszy (ostatnia data 35–50 tys.).
Luzowanie limitów c_LT/W_max nie było potrzebne.

Średni zrealizowany CE [% rocznie] po 7 modelach (hold-out; kalibracja w nawiasie):

| profil | ce_dep | ce | maxret_es | min_es | ew | mv_hist |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 3,76 (5,32) | 4,32 (4,77) | 2,29 (3,42) | −0,23 (1,58) | 3,52 (3,30) | 1,85 (5,13) |
| P2 | 5,79 (9,00) | 6,19 (8,54) | 4,15 (7,23) | 0,19 (3,25) | 4,56 (5,25) | 4,39 (7,45) |
| P3 | 6,87 (11,86) | 7,03 (11,69) | 5,06 (8,88) | 0,81 (4,32) | 5,26 (6,58) | 5,18 (11,13) |
| P4 | 9,96 (15,83) | 10,93 (16,05) | 6,12 (10,22) | 1,39 (5,11) | 5,96 (7,95) | 8,21 (15,06) |
| P5 | 15,04 (21,90) | 15,43 (21,54) | 6,84 (12,00) | 1,58 (5,78) | 6,49 (9,00) | 12,11 (17,56) |

Strategie CE dominują pozostałe reguły w obu próbach. Kara za zależność podnosi CE w kalibracji
(P1–P3, P5), ale nie w hold-out. MCS obejmuje 24–27 z 30 szeregów każdego profilu w hold-out, a test
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

- **Asynchroniczność.** GPW zamyka się o 17:00, NYSE o 22:00, a fixing NBP jest ok. 11:50. Dzienna
  zależność między rynkami jest więc zaniżona, a przeliczenie na PLN używa kursu z innej chwili niż cena.
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
- **Stały horyzont.** Horyzont scenariuszy H = 21 sesji jest stały, a okres realizacji ma 19–23 sesje.
- **Zależność ścieżek w bootstrapie.** W bootstrapie stacjonarnym sąsiednie ścieżki 21-dniowe z jednej
  replikacji są słabo zależne, więc SE jest lekko zaniżony.
