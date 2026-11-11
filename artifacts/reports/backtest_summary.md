# Backtest – podsumowanie

Przebieg `3ababedef5d8`; daty rebalansowania 2008-01-02 – 2026-10-01 (226), w tym okresy realizacji: kalibracja 156 mies., hold-out 69 mies. Opóźnienie wykonania 1 sesja, koszt 0.1% wartości transakcji, r_f – WIBOR 1M.
Szeregi bazowe: 232 (7 modeli × 4 strategie × 5 profili + 7 modeli × 4 strategie × 3 profile z nakładką opcyjną `_ov` + ew + idx_spx + idx_wig20 + 5 × mv_hist).
Scenariusze: mediana N = 12500, maks. 50000; odsetek par (data, model) z N = n_max: 0.2%.
Czas: średnio 721 s/datę (1 proces), łącznie 45.2 h CPU.
Luzowanie c_LT: 0.00% problemów poziomu 2; W_max: 0.00%; maxret_es bez rozwiązania przy ES ≤ ES(EW) (min ES): 0.00%.

## Najlepsze szeregi wg CE – próba kalibracyjna

| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |
|---|---|---|---|---:|---|---:|
| P1 | cvine/ce_dep (2.32) | cvine/ce (2.09) | empirical/ce_dep (2.05) | 0.81 | 1.39 (6.) | 32 |
| P2 | cvine/ce_dep (3.63) | cvine/ce (3.36) | empirical/ce_dep (3.34) | 2.50 | 2.84 (6.) | 30 |
| P3 | cvine/ce_ov (8.14) | rvine/ce_ov (7.42) | gauss/maxret_es_ov (7.11) | 3.64 | 4.39 (26.) | 59 |
| P4 | cvine/ce_ov (12.09) | cvine/ce_dep_ov (11.67) | cvine/ce (11.02) | 4.80 | 8.27 (22.) | 50 |
| P5 | cvine/ce_ov (18.27) | indep/ce_ov (17.49) | cvine/ce (17.43) | 5.69 | 14.00 (24.) | 33 |

## Najlepsze szeregi wg CE – próba hold-out

| profil | 1. | 2. | 3. | ew (CE) | rvine/ce_dep (CE, miejsce) | w MCS |
|---|---|---|---|---:|---|---:|
| P1 | student_t/ce_dep (3.70) | gauss/ce_dep (3.56) | cvine/min_es (3.11) | 3.08 | 2.53 (6.) | 32 |
| P2 | idx_spx (5.20) | ew (4.09) | gauss/ce_dep (4.06) | 4.09 | 3.85 (6.) | 32 |
| P3 | idx_spx (7.03) | gauss/ce_dep_ov (6.65) | student_t/ce_dep_ov (6.06) | 4.77 | 3.30 (31.) | 59 |
| P4 | gauss/ce_dep_ov (10.00) | cvine/maxret_es_ov (9.94) | gauss/ce_ov (9.88) | 5.45 | 3.48 (46.) | 49 |
| P5 | gauss/ce_dep_ov (20.46) | gauss/ce_ov (19.85) | student_t/ce_dep_ov (17.40) | 5.96 | 10.92 (31.) | 51 |

## Testy VaR/ES (szeregi modelowe, odsetek bez odrzucenia na poziomie 5%)

| próba | T (mies.) | oczekiwane przekr. | śr. przekr. | Kupiec | Christoffersen cc | Z₂ | DE uc | DE cc(5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kalibracyjna | 156 | 3.9 | 14.7 | 42% | 42% | 27% | 32% | 32% |
| hold-out | 69 | 1.7 | 6.1 | 61% | 74% | 58% | 57% | 43% |

Przy próbie miesięcznej oczekiwana liczba przekroczeń jest mała, więc moc testów jest niska.

## Wrażliwość (próba kalibracyjna, strategia ce_dep)

CE zrealizowany [% rocznie] przy γ profilu; wartości średnie po 7 modelach zależności (kolumna base – parametry wyjściowe).

| profil | base | lag0 | q10 | rhoS |
|---|---:|---:|---:|---:|
| P1 | 0.71 | 2.04 | 0.70 | 0.46 |
| P2 | 2.55 | 3.52 | 2.50 | 2.30 |
| P3 | 4.29 | 4.95 | 4.25 | 4.45 |
| P4 | 8.57 | 9.06 | 8.57 | 8.59 |
| P5 | 14.26 | 14.75 | 14.28 | 14.15 |

Siatka κ (CE [% rocznie], średnia po modelach) i wybór κ maksymalizujący Σ CE przy ograniczeniu κ_P1 ≥ … ≥ κ_P5 (informacyjnie – hold-out liczony z κ wyjściowymi):

| profil | κ = 0 | κ = 0.05 | κ = 0.15 | κ = 0.35 | κ = 0.6 | κ = 1 | κ = 2 | κ wyjściowe | κ wybrane |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0.10 | 0.11 | 0.21 | 0.35 | 0.50 | 0.71 | 1.32 | 1 | 2 |
| P2 | 1.91 | 1.91 | 1.94 | 2.21 | 2.55 | 3.19 | 3.69 | 0.6 | 2 |
| P3 | 4.73 | 4.70 | 4.66 | 4.29 | 4.81 | 5.18 | 5.24 | 0.35 | 2 |
| P4 | 8.59 | 8.69 | 8.57 | 8.45 | 8.42 | 7.97 | 6.93 | 0.15 | 0.05 |
| P5 | 14.66 | 14.26 | 13.66 | 13.46 | 13.11 | 11.91 | 8.22 | 0.05 | 0 |

Szczegóły: `ranking_calibration.md`, `ranking_holdout.md`, `dependence_models.md`, `es_mc.md`; wykresy: `figures/backtest/`.

