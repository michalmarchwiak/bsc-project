# Model zależności: R-vine i comiesięczne rebalansowanie

## Sub-fundusze

`config/subfunds.yaml` – 9 sub-funduszy zgodnych z grupami tickerów w `data_import.py`:
SF1 akcje polskie, SF2 akcje USA, SF3 rynki rozwinięte ex-USA, SF4 rynki wschodzące,
SF5 obligacje skarbowe USA, SF6 obligacje korporacyjne IG, SF7 high yield, SF8 złoto
i spółki wydobywcze, SF9 REIT. Wewnątrz sub-funduszu wagi są równe między aktywami
dostępnymi w dniu rebalansowania (macierz look-through A, x = AᵀW).

## Model

1. **Marginesy** (`pf.models.margins`): AR(1)–GJR-GARCH(1,1) z rozkładem skośnym t Hansena,
   estymowane na dziennych log-stopach w PLN (`arch`). Reszty standaryzowane ẑ_t tworzą panel:
   dni zamknięcia całego rynku (GPW lub USA) są usuwane, pojedyncze braki → ẑ = 0.
2. **Pseudo-obserwacje**: rangi reszt / (n + 1) – estymacja kopuły półparametryczna.
3. **R-vine** (`pf.models.rvine`, `pf.models.copulas`): struktura algorytmem Dißmanna
   (maksymalne drzewa rozpinające z wagami |τ̂ Kendalla|, warunek bliskości), obcięta na
   3 drzewach (wyższe drzewa – kopuła niezależności). Dla każdej krawędzi: test niezależności
   (τ Kendalla, α = 5%), parametr z odwrócenia τ (ρ = sin(πτ/2), θ_Clayton = 2τ/(1−τ),
   θ_Gumbel = 1/(1−τ); ν kopuły t z MNW przy ustalonym ρ), rodzina z kryterium AIC spośród:
   Gauss, t, Clayton, Gumbel oraz obrotów 180° (τ > 0, ogon przeciwny) i 90°/270° (τ < 0).
4. **Scenariusze** (`pf.sim.scenarios`): H = 21 dziennych losowań z R-vine na scenariusz,
   kwantyle skew-t, rekursja GARCH od stanu na koniec okna, proste stopy zwrotu aktywów
   w horyzoncie, agregacja do sub-funduszy.
5. **Skład sub-funduszy** (`pf.opt.cvar`, `pipelines.monthly_rebalance.optimize_subfund`):
   dla każdego sub-funduszu k osobny LP Rockafellara–Uryaseva na scenariuszach jego aktywów:
   max E[R_kᵀw] przy ES_97,5%(−R_kᵀw) ≤ ES_97,5% portfela równych wag tego sub-funduszu,
   Σw = 1, 0 ≤ w_i ≤ max(30%, 1/n_k). Portfel równych wag jest zawsze dopuszczalny, więc skład
   optymalny ma nie wyższe ES i nie niższy oczekiwany zwrot (na scenariuszach). Wagi między
   sub-funduszami (W) nie są wyznaczane – zapisywane są scenariusze stóp zwrotu sub-funduszy
   (`scenarios_latest.parquet`), na których W dobierane jest dla profilu ryzyka.

## Uruchomienie

```bash
python -m pipelines.fit_rvine_full_sample      # pełna próba → artifacts/reports/rvine_full_sample.md
python -m pipelines.monthly_rebalance --jobs 4 # wszystkie brakujące daty rebalansowania
python -m pipelines.monthly_rebalance --date 2020-12-01   # pojedyncza data
```

Rebalansowanie odbywa się w pierwszej sesji miesiąca t na danych do sesji t−1 (okno 1260
sesji, aktywa z ≥ 750 obserwacjami w oknie; pierwsza data, gdy każdy sub-fundusz ma
dostępne aktywo). Przebieg jest przyrostowy – każde kolejne uruchomienie liczy tylko nowe
daty, więc wystarczy uruchamiać je po pierwszej sesji miesiąca (po odświeżeniu danych).
Modele zapisane dla danej daty (R-vine, parametry GARCH) są przy ponownym liczeniu wczytywane
(`--refit` estymuje od nowa), a scenariusze są powtarzalne (ziarno zależne od daty).
Wyniki: `artifacts/rebalance/` – `composition.parquet` (skład: date, subfund, ticker, weight,
weight_ew), `subfunds.parquet` (E[R], ES, VaR: skład optymalny i równe wagi), `models.parquet`,
`realized.parquet`, `turnover.parquet`, R-vine i parametry GARCH dla każdej daty, scenariusze sub-funduszy najnowszej
daty; raport `artifacts/reports/rebalance_summary.md` (każdy sub-fundusz: skład optymalny vs równe wagi).
Parametry: `config/models.yaml`.

## Ograniczenia

- Notowania GPW i NYSE nie są synchroniczne (zamknięcie 17:00 vs 22:00 czasu polskiego),
  więc dzienna zależność między rynkami jest zaniżona; zależność wewnątrz rynku nie.
- Stopy w PLN zawierają czynnik USD/PLN – dla aktywów USA o niskiej zmienności (krótkie
  obligacje skarbowe) zależność wynika głównie z kursu walutowego (τ ≈ 0,96 dla SHY–SCHO).
- Duplikaty ekspozycji (GLD/IAU/SGOL, SHY/VGSH/SCHO) dają τ > 0,9; parametry kopuł są
  ograniczone (|ρ| ≤ 0,9999, θ_Clayton ≤ 50, θ_Gumbel ≤ 30).
- Parametry z odwrócenia τ nie są estymatorami MNW (wyjątek: ν kopuły t); AIC liczony jest
  w tych punktach.
- Innowacje kopuły są iid w czasie (zależność stała w oknie estymacji); zmienność warunkowa
  pochodzi wyłącznie z marginesów GARCH.
- Oczekiwane stopy zwrotu wynikają ze stałych AR(1) estymowanych w oknie – są obarczone
  dużym błędem, a LP na nie reaguje silnie; limit 30% na ticker ogranicza koncentrację.
- W sub-funduszu z mniej niż 4 dostępnymi aktywami (np. SF7 przed notowaniem ANGL, FALN)
  limit 1/n wymusza równe wagi.
- Obrót w dniu t: TO_t = ½ Σ_i |w_{i,t} − w̃_{i,t}|, gdzie w̃_{i,t} = w_{i,t−1}(1 + r_i) / Σ_j w_{j,t−1}(1 + r_j)
  – wagi z poprzedniego rebalansowania po dryfie cen w okresie kupna i trzymania (także dla równych
  wag, które po dryfie trzeba przywracać). Dryf nie zmienia zrealizowanych stóp zwrotu brutto.
- Koszty transakcyjne (opcjonalne, `transaction_cost`, domyślnie 0,1% wartości transakcji;
  `--cost 0` wyłącza): k_t = c·Σ_i |w_{i,t} − w̃_{i,t}| = 2c·TO_t, stopa netto okresu
  r_net = (1 − k_t)(1 + r) − 1; przy zakupie początkowym k = c. Koszty nie wchodzą do
  optymalizacji (LP nie karze obrotu) – obniżają jedynie zrealizowane stopy zwrotu.
