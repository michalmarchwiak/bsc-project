# Backtest – podsumowanie

Przebieg `4010467a06e7`; daty rebalansowania 2008-01-02 – 2026-10-01 (226), w tym okresy realizacji: kalibracja 156 mies., hold-out 69 mies. Opóźnienie wykonania 1 sesja, koszt 0.1% wartości transakcji, r_f – WIBOR 1M.
Szeregi bazowe: 146 (7 modeli × 4 strategie × 5 profili + ew + 5 × mv_hist).
Scenariusze: mediana N = 12500, maks. 50000; odsetek par (data, model) z N = n_max: 0.1%.
Czas: średnio 671 s/datę (1 proces), łącznie 42.1 h CPU.
Luzowanie c_LT: 0.00% problemów poziomu 2; W_max: 0.00%; maxret_es bez rozwiązania przy ES ≤ ES(EW) (min ES): 0.00%.

## Najlepsze szeregi wg CE – próba kalibracyjna

| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |
|---|---|---|---|---:|---|---:|
| P1 | gauss/ce_dep (6.11) | empirical/ce_dep (6.02) | cvine/ce_dep (5.96) | 3.30 | 5.76 (6.) | 23 |
| P2 | gauss/ce_dep (9.87) | student_t/ce_dep (9.81) | empirical/ce_dep (9.59) | 5.25 | 9.35 (4.) | 24 |
| P3 | gauss/ce_dep (12.33) | student_t/ce_dep (12.28) | rvine/ce_dep (12.26) | 6.58 | 12.26 (3.) | 29 |
| P4 | indep/ce (17.51) | indep/ce_dep (17.07) | cvine/ce (16.76) | 7.95 | 15.46 (10.) | 30 |
| P5 | cvine/ce (23.21) | cvine/ce_dep (23.14) | indep/ce_dep (23.04) | 9.00 | 21.95 (5.) | 30 |

## Najlepsze szeregi wg CE – próba hold-out

| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |
|---|---|---|---|---:|---|---:|
| P1 | cvine/ce (6.43) | empirical/ce (5.20) | gauss/ce (4.88) | 3.52 | 4.46 (5.) | 24 |
| P2 | cvine/ce (7.38) | empirical/ce (7.18) | gauss/ce (6.96) | 4.56 | 6.33 (6.) | 26 |
| P3 | empirical/ce (7.62) | cvine/ce (7.61) | empirical/ce_dep (7.53) | 5.26 | 6.82 (10.) | 26 |
| P4 | dvine/ce (11.73) | empirical/ce (11.52) | cvine/ce (11.06) | 5.96 | 9.86 (11.) | 27 |
| P5 | rvine/ce (15.94) | empirical/ce (15.63) | cvine/ce (15.61) | 6.49 | 15.29 (6.) | 26 |

## Testy VaR/ES (szeregi modelowe, odsetek bez odrzucenia na poziomie 5%)

| próba | T (mies.) | oczekiwane przekr. | śr. przekr. | Kupiec | Christoffersen cc | Z₂ | DE uc | DE cc(5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kalibracyjna | 156 | 3.9 | 11.4 | 57% | 53% | 42% | 46% | 45% |
| hold-out | 69 | 1.7 | 4.2 | 82% | 85% | 80% | 84% | 19% |

Przy próbie miesięcznej oczekiwana liczba przekroczeń jest mała, więc moc testów jest niska.

## Wrażliwość (próba kalibracyjna, strategia ce_dep)

CE zrealizowany [% rocznie] przy γ profilu; wartości średnie po 7 modelach zależności (kolumna base – parametry wyjściowe).

| profil | base | lag0 | q10 | rhoS |
|---|---:|---:|---:|---:|
| P1 | 5.32 | 6.54 | 5.42 | 5.66 |
| P2 | 9.00 | 10.13 | 9.11 | 9.13 |
| P3 | 11.86 | 12.89 | 11.98 | 12.39 |
| P4 | 15.83 | 16.58 | 15.90 | 16.25 |
| P5 | 21.90 | 22.69 | 21.88 | 21.90 |

Siatka κ (CE [% rocznie], średnia po modelach) i wybór κ maksymalizujący Σ CE przy ograniczeniu κ_P1 ≥ … ≥ κ_P5 (informacyjnie – hold-out liczony z κ wyjściowymi):

| profil | κ = 0 | κ = 0.05 | κ = 0.15 | κ = 0.35 | κ = 0.6 | κ = 1 | κ = 2 | κ wyjściowe | κ wybrane |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 4.77 | 4.89 | 5.03 | 5.21 | 5.35 | 5.32 | 5.22 | 1 | 0.6 |
| P2 | 8.54 | 8.65 | 8.87 | 9.03 | 9.00 | 8.72 | 8.39 | 0.6 | 0.35 |
| P3 | 11.69 | 11.75 | 11.75 | 11.86 | 11.41 | 11.10 | 10.60 | 0.35 | 0.35 |
| P4 | 16.05 | 16.04 | 15.83 | 15.92 | 15.73 | 14.81 | 13.20 | 0.15 | 0.05 |
| P5 | 21.54 | 21.90 | 21.53 | 21.63 | 20.66 | 18.93 | 15.67 | 0.05 | 0.05 |

Szczegóły: `ranking_calibration.md`, `ranking_holdout.md`, `dependence_models.md`, `es_mc.md`; wykresy: `figures/backtest/`.

