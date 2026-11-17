# Koszt kroku modelowego backtestu i prognoza czasu (Etap 2)

Scenariusze: partie 2500, N ∈ [5000, 50000], stop przy max_k SE/|ÊS| ≤ 0.02 (portfele kontrolne: EW w każdym SF i EW między SF), opóźnienie wykonania 1 sesja (H + L dni symulacji).
Pomiar na jednym rdzeniu na datę (jobs = 1 w estymacji GARCH, wątki BLAS = 1).

## Czasy kroku modelowego [s]

| data | aktywa | panel | marginesy | fit rvine | fit cvine | fit dvine | fit gauss | fit student_t | fit indep | fit empirical | sym. rvine | sym. cvine | sym. dvine | sym. gauss | sym. student_t | sym. indep | sym. empirical | log-score | razem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2010-05-03 | 65 | 783 | 1.7 | 5.2 | 5.7 | 5.2 | 0.0 | 0.2 | 0.0 | 0.0 | 20.9 | 8.4 | 22.6 | 3.2 | 4.4 | 2.9 | 2.9 | 0.0 | 83.5 |
| 2018-08-01 | 84 | 1255 | 2.0 | 10.9 | 10.9 | 10.3 | 0.0 | 0.4 | 0.0 | 0.0 | 29.0 | 19.1 | 34.6 | 4.0 | 5.5 | 6.5 | 4.6 | 0.0 | 137.7 |
| 2026-10-01 | 86 | 1255 | 2.1 | 11.9 | 12.0 | 10.5 | 0.0 | 0.4 | 0.0 | 0.0 | 28.6 | 15.0 | 33.1 | 4.1 | 7.2 | 8.5 | 3.8 | 0.0 | 137.3 |

## Liczba scenariuszy N i względny błąd ÊS (max po portfelach kontrolnych)

| data | rvine | cvine | dvine | gauss | student_t | indep | empirical |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2010-05-03 | 15000 (2.0%) | 12500 (1.8%) | 12500 (1.8%) | 10000 (1.9%) | 10000 (2.0%) | 10000 (1.9%) | 10000 (1.9%) |
| 2018-08-01 | 12500 (1.9%) | 12500 (2.0%) | 12500 (2.0%) | 10000 (2.0%) | 10000 (1.9%) | 17500 (2.0%) | 12500 (1.8%) |
| 2026-10-01 | 12500 (1.8%) | 10000 (1.9%) | 12500 (1.8%) | 10000 (2.0%) | 12500 (1.8%) | 22500 (1.9%) | 10000 (1.9%) |

Portfel wyznaczający N (max SE/|ÊS|): 2010-05-03/cvine: SF4, 2010-05-03/dvine: SF4, 2010-05-03/empirical: SF4, 2010-05-03/gauss: SF3, 2010-05-03/indep: EW, 2010-05-03/rvine: SF4, 2010-05-03/student_t: SF4, 2018-08-01/cvine: SF3, 2018-08-01/dvine: SF3, 2018-08-01/empirical: SF3, 2018-08-01/gauss: SF9, 2018-08-01/indep: EW, 2018-08-01/rvine: SF2, 2018-08-01/student_t: SF9, 2026-10-01/cvine: SF1, 2026-10-01/dvine: SF7, 2026-10-01/empirical: SF3, 2026-10-01/gauss: SF3, 2026-10-01/indep: EW, 2026-10-01/rvine: SF3, 2026-10-01/student_t: SF3.

## Mikro-benchmark optymalizacji [s na problem]

| aktywa | N | LP max E[R] przy ES ≤ L | LP min ES | SLSQP CE − κD |
|---:|---:|---:|---:|---:|
| 10 | 5000 | 0.09 | 0.08 | 0.002 |
| 10 | 10000 | 0.28 | 0.27 | 0.002 |
| 10 | 20000 | 0.98 | 0.89 | 0.002 |
| 10 | 50000 | 6.12 | 4.88 | 0.006 |

Średnie N ≈ 12262; koszt optymalizacji na datę ≈ 7 modeli × 5 profili × 11 problemów (10 SF + poziom 2) × (2 LP + 2 SLSQP) ≈ 305 s (czasy interpolowane log-log przy średnim N; LP 0.79 s, SLSQP 0.004 s na parę problemów; SLSQP to prototyp funkcji celu z Etapu 3). Czas pełnego LP rośnie z N w przybliżeniu kwadratowo.

## Prognoza czasu pełnego przebiegu

Krok modelowy: średnio 119 s/datę; z optymalizacją ≈ 424 s/datę.

| próba | liczba dat | jobs = 1 [h] | jobs = 4 [h] | jobs = 8 [h] | jobs = 16 [h] |
|---|---:|---:|---:|---:|---:|
| obecny plik danych | 198 | 23.3 | 5.8 | 2.9 | 1.5 |
| docelowa (kalibracja + hold-out) | 226 | 26.6 | 6.7 | 3.3 | 1.7 |

Uwagi: czas symulacji rośnie liniowo z N i (dla vine'ów) z liczbą krawędzi z kopułą t; daty wcześniejsze mają mniej aktywów. Prognoza dla próby docelowej zakłada koszt daty jak w pomiarze (≈ 80 aktywów). Analizy wrażliwości (Etap 4) i MC ex post (Etap 5) nie są wliczone.
