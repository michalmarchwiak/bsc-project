# Backtest – podsumowanie

Przebieg `c3ec64d93522`; daty rebalansowania 2008-01-02 – 2026-10-01 (226), w tym okresy realizacji: kalibracja 156 mies., hold-out 69 mies. Opóźnienie wykonania 1 sesja, koszt 0.1% wartości transakcji, r_f – WIBOR 1M.
Szeregi bazowe: 232 (7 modeli × 4 strategie × 5 profili + 7 modeli × 4 strategie × 3 profile z nakładką opcyjną `_ov` + ew + idx_spx + idx_wig20 + 5 × mv_hist).
Scenariusze: mediana N = 12500, maks. 50000; odsetek par (data, model) z N = n_max: 0.2%.
Czas: średnio 786 s/datę (1 proces), łącznie 49.4 h CPU.
Luzowanie c_LT: 0.00% problemów poziomu 2; W_max: 0.00%; maxret_es bez rozwiązania przy ES ≤ ES(EW) (min ES): 0.00%.

## Najlepsze szeregi wg CE – próba kalibracyjna

| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |
|---|---|---|---|---:|---|---:|
| P1 | cvine/ce_dep (2.65) | empirical/ce_dep (2.31) | gauss/ce_dep (2.15) | 0.81 | 1.69 (8.) | 32 |
| P2 | cvine/ce_dep (4.29) | empirical/ce_dep (3.90) | gauss/maxret_es (3.38) | 2.50 | 3.03 (8.) | 31 |
| P3 | gauss/ce_ov (5.99) | rvine/ce_ov (5.89) | student_t/ce_ov (5.84) | 3.64 | 4.17 (28.) | 58 |
| P4 | cvine/ce_ov (11.29) | cvine/ce (11.02) | cvine/ce_dep_ov (10.85) | 4.80 | 8.27 (19.) | 49 |
| P5 | cvine/ce (17.43) | cvine/ce_ov (17.08) | cvine/ce_dep (16.65) | 5.69 | 14.00 (22.) | 33 |

## Najlepsze szeregi wg CE – próba hold-out

| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |
|---|---|---|---|---:|---|---:|
| P1 | cvine/ce (4.11) | ew (3.08) | cvine/min_es (3.08) | 3.08 | 2.50 (10.) | 28 |
| P2 | cvine/ce (6.15) | idx_spx (5.20) | indep/ce (4.80) | 4.09 | 3.75 (14.) | 30 |
| P3 | empirical/ce_ov (7.77) | cvine/ce (7.30) | empirical/ce (7.23) | 4.77 | 5.44 (19.) | 56 |
| P4 | cvine/maxret_es_ov (9.20) | gauss/ce_dep_ov (9.05) | gauss/ce_ov (8.97) | 5.45 | 3.48 (45.) | 49 |
| P5 | gauss/ce_dep_ov (18.92) | gauss/ce_ov (18.32) | gauss/ce (17.30) | 5.96 | 10.92 (31.) | 49 |

## Testy VaR/ES (szeregi modelowe, odsetek bez odrzucenia na poziomie 5%)

| próba | T (mies.) | oczekiwane przekr. | śr. przekr. | Kupiec | Christoffersen cc | Z₂ | DE uc | DE cc(5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kalibracyjna | 156 | 3.9 | 14.5 | 48% | 48% | 33% | 38% | 25% |
| hold-out | 69 | 1.7 | 6.1 | 62% | 72% | 60% | 60% | 37% |

Przy próbie miesięcznej oczekiwana liczba przekroczeń jest mała, więc moc testów jest niska.

## Wrażliwość (próba kalibracyjna, strategia ce_dep)

CE zrealizowany [% rocznie] przy γ profilu; wartości średnie po 7 modelach zależności (kolumna base – parametry wyjściowe).

| profil | base | lag0 | q10 | rhoS |
|---|---:|---:|---:|---:|
| P1 | 1.23 | 2.32 | 1.25 | 1.23 |
| P2 | 2.98 | 3.73 | 3.08 | 3.03 |
| P3 | 4.27 | 4.86 | 4.31 | 4.63 |
| P4 | 8.57 | 9.06 | 8.57 | 8.59 |
| P5 | 14.26 | 14.75 | 14.28 | 14.15 |

Siatka κ (CE [% rocznie], średnia po modelach) i wybór κ maksymalizujący Σ CE przy ograniczeniu κ_P1 ≥ … ≥ κ_P5 (informacyjnie – hold-out liczony z κ wyjściowymi):

| profil | κ = 0 | κ = 0.05 | κ = 0.15 | κ = 0.35 | κ = 0.6 | κ = 1 | κ = 2 | κ wyjściowe | κ wybrane |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0.42 | 0.53 | 0.66 | 0.87 | 1.07 | 1.23 | 1.47 | 1 | 2 |
| P2 | 2.31 | 2.34 | 2.37 | 2.63 | 2.98 | 3.27 | 3.44 | 0.6 | 2 |
| P3 | 4.40 | 4.40 | 4.38 | 4.27 | 4.52 | 4.77 | 5.01 | 0.35 | 2 |
| P4 | 8.59 | 8.69 | 8.57 | 8.45 | 8.42 | 7.97 | 6.93 | 0.15 | 0.05 |
| P5 | 14.66 | 14.26 | 13.66 | 13.46 | 13.11 | 11.91 | 8.22 | 0.05 | 0 |

Szczegóły: `ranking_calibration.md`, `ranking_holdout.md`, `dependence_models.md`, `es_mc.md`; wykresy: `figures/backtest/`.

