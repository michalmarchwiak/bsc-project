# Backtest – podsumowanie

Przebieg `83254f06e339`; daty rebalansowania 2008-01-02 – 2026-10-01 (226), w tym okresy realizacji: kalibracja 156 mies., hold-out 69 mies. Opóźnienie wykonania 1 sesja, koszt 0.1% wartości transakcji, r_f – WIBOR 1M.
Szeregi bazowe: 230 (7 modeli × 4 strategie × 5 profili + 7 modeli × 4 strategie × 3 profile z nakładką opcyjną `_ov` + ew + 5 × mv_hist).
Scenariusze: mediana N = 12500, maks. 50000; odsetek par (data, model) z N = n_max: 0.2%.
Czas: średnio 838 s/datę (1 proces), łącznie 52.6 h CPU.
Luzowanie c_LT: 0.00% problemów poziomu 2; W_max: 0.00%; maxret_es bez rozwiązania przy ES ≤ ES(EW) (min ES): 0.00%.

## Najlepsze szeregi wg CE – próba kalibracyjna

| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |
|---|---|---|---|---:|---|---:|
| P1 | student_t/ce_dep (6.67) | gauss/ce_dep (6.64) | empirical/ce_dep (6.24) | 3.65 | 5.85 (6.) | 21 |
| P2 | gauss/ce_dep (8.87) | student_t/ce_dep (8.79) | gauss/ce (8.48) | 5.23 | 8.26 (5.) | 27 |
| P3 | gauss/ce_dep_ov (12.82) | indep/ce_ov (12.80) | student_t/ce_ov (12.80) | 6.30 | 10.39 (31.) | 38 |
| P4 | gauss/ce_dep_ov (18.89) | student_t/ce_dep_ov (18.78) | gauss/ce_ov (18.31) | 7.39 | 14.98 (27.) | 31 |
| P5 | student_t/ce_dep_ov (25.41) | gauss/ce_dep_ov (25.36) | gauss/ce_ov (25.05) | 8.23 | 21.87 (18.) | 31 |

## Najlepsze szeregi wg CE – próba hold-out

| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |
|---|---|---|---|---:|---|---:|
| P1 | gauss/ce_dep (5.33) | student_t/ce_dep (5.20) | rvine/ce_dep (5.10) | 3.52 | 5.10 (3.) | 30 |
| P2 | gauss/ce (7.16) | student_t/ce (7.04) | cvine/ce (7.02) | 4.37 | 5.49 (10.) | 30 |
| P3 | gauss/ce_dep_ov (10.15) | student_t/ce_dep_ov (9.54) | gauss/ce_dep (9.41) | 4.93 | 8.02 (7.) | 54 |
| P4 | gauss/ce_ov (12.53) | gauss/ce_dep_ov (12.51) | gauss/ce_dep (11.93) | 5.50 | 9.86 (25.) | 55 |
| P5 | gauss/ce_ov (20.76) | student_t/ce_dep_ov (20.42) | gauss/ce_dep_ov (20.41) | 5.92 | 14.98 (24.) | 34 |

## Testy VaR/ES (szeregi modelowe, odsetek bez odrzucenia na poziomie 5%)

| próba | T (mies.) | oczekiwane przekr. | śr. przekr. | Kupiec | Christoffersen cc | Z₂ | DE uc | DE cc(5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kalibracyjna | 156 | 3.9 | 13.4 | 46% | 48% | 29% | 32% | 42% |
| hold-out | 69 | 1.7 | 4.9 | 80% | 80% | 79% | 79% | 22% |

Przy próbie miesięcznej oczekiwana liczba przekroczeń jest mała, więc moc testów jest niska.

## Wrażliwość (próba kalibracyjna, strategia ce_dep)

CE zrealizowany [% rocznie] przy γ profilu; wartości średnie po 7 modelach zależności (kolumna base – parametry wyjściowe).

| profil | base | lag0 | q10 | rhoS |
|---|---:|---:|---:|---:|
| P1 | 5.54 | 6.53 | 5.57 | 5.69 |
| P2 | 8.06 | 8.95 | 8.01 | 8.50 |
| P3 | 10.99 | 11.63 | 10.93 | 11.73 |
| P4 | 16.13 | 16.82 | 16.17 | 16.43 |
| P5 | 22.19 | 23.01 | 22.19 | 22.25 |

Siatka κ (CE [% rocznie], średnia po modelach) i wybór κ maksymalizujący Σ CE przy ograniczeniu κ_P1 ≥ … ≥ κ_P5 (informacyjnie – hold-out liczony z κ wyjściowymi):

| profil | κ = 0 | κ = 0.05 | κ = 0.15 | κ = 0.35 | κ = 0.6 | κ = 1 | κ = 2 | κ wyjściowe | κ wybrane |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 5.04 | 5.18 | 5.36 | 5.49 | 5.57 | 5.54 | 5.60 | 1 | 2 |
| P2 | 7.76 | 7.69 | 7.63 | 7.95 | 8.06 | 7.80 | 7.64 | 0.6 | 0.6 |
| P3 | 11.39 | 11.47 | 11.48 | 10.99 | 10.49 | 9.76 | 8.81 | 0.35 | 0.15 |
| P4 | 16.02 | 16.24 | 16.13 | 16.05 | 14.62 | 12.77 | 10.21 | 0.15 | 0.05 |
| P5 | 21.95 | 22.19 | 22.04 | 22.03 | 20.42 | 16.28 | 10.57 | 0.05 | 0.05 |

Szczegóły: `ranking_calibration.md`, `ranking_holdout.md`, `dependence_models.md`, `es_mc.md`; wykresy: `figures/backtest/`.

