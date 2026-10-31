# Koszt kroku modelowego backtestu i prognoza czasu (Etap 2)

Scenariusze: partie 2500, N ∈ [5000, 50000], stop przy max_k SE/|ÊS| ≤ 0.02 (portfele kontrolne: EW w każdym SF i EW między SF), opóźnienie wykonania 1 sesja (H + L dni symulacji).
Pomiar na jednym rdzeniu na datę (jobs = 1 w estymacji GARCH, wątki BLAS = 1).

## Czasy kroku modelowego [s]

| data | aktywa | panel | marginesy | fit rvine | fit cvine | fit dvine | fit gauss | fit student_t | fit indep | fit empirical | sym. rvine | sym. cvine | sym. dvine | sym. gauss | sym. student_t | sym. indep | sym. empirical | log-score | razem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2014-02-03 | 73 | 744 | 2.6 | 10.1 | 9.9 | 10.0 | 0.0 | 0.3 | 0.0 | 0.0 | 43.5 | 23.4 | 40.4 | 7.6 | 7.9 | 8.7 | 5.4 | 0.1 | 170.1 |
| 2017-07-03 | 77 | 1184 | 3.1 | 16.4 | 16.5 | 15.7 | 0.0 | 0.6 | 0.0 | 0.0 | 51.9 | 33.8 | 61.9 | 8.0 | 12.9 | 10.3 | 5.9 | 0.1 | 237.1 |
| 2020-12-01 | 79 | 1186 | 2.9 | 17.6 | 17.1 | 16.7 | 0.0 | 0.6 | 0.0 | 0.0 | 81.1 | 42.2 | 81.3 | 7.5 | 12.7 | 8.4 | 7.4 | 0.1 | 295.7 |

## Liczba scenariuszy N i względny błąd ÊS (max po portfelach kontrolnych)

| data | rvine | cvine | dvine | gauss | student_t | indep | empirical |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2014-02-03 | 12500 (1.8%) | 10000 (2.0%) | 10000 (2.0%) | 12500 (1.9%) | 10000 (1.9%) | 15000 (1.9%) | 10000 (2.0%) |
| 2017-07-03 | 12500 (1.9%) | 12500 (1.7%) | 12500 (1.9%) | 12500 (1.9%) | 15000 (1.9%) | 17500 (2.0%) | 10000 (2.0%) |
| 2020-12-01 | 17500 (1.9%) | 15000 (1.8%) | 15000 (1.9%) | 12500 (1.9%) | 15000 (1.8%) | 15000 (1.9%) | 12500 (1.8%) |

Portfel wyznaczający N (max SE/|ÊS|): 2014-02-03/cvine: SF3, 2014-02-03/dvine: SF7, 2014-02-03/empirical: SF9, 2014-02-03/gauss: SF3, 2014-02-03/indep: SF9, 2014-02-03/rvine: SF3, 2014-02-03/student_t: SF3, 2017-07-03/cvine: SF9, 2017-07-03/dvine: SF2, 2017-07-03/empirical: SF3, 2017-07-03/gauss: SF3, 2017-07-03/indep: EW, 2017-07-03/rvine: SF2, 2017-07-03/student_t: SF3, 2020-12-01/cvine: SF3, 2020-12-01/dvine: SF2, 2020-12-01/empirical: SF7, 2020-12-01/gauss: SF3, 2020-12-01/indep: EW, 2020-12-01/rvine: SF2, 2020-12-01/student_t: SF3.

## Mikro-benchmark optymalizacji [s na problem]

| aktywa | N | LP max E[R] przy ES ≤ L | LP min ES | SLSQP CE − κD |
|---:|---:|---:|---:|---:|
| 10 | 5000 | 0.14 | 0.12 | 0.005 |
| 10 | 10000 | 0.46 | 0.44 | 0.004 |
| 10 | 20000 | 1.66 | 1.57 | 0.007 |
| 10 | 50000 | 10.55 | 8.78 | 0.011 |

Średnie N ≈ 13095; koszt optymalizacji na datę ≈ 7 modeli × 5 profili × 10 problemów (9 SF + poziom 2) × (2 LP + 2 SLSQP) ≈ 520 s (czasy interpolowane log-log przy średnim N; LP 1.48 s, SLSQP 0.009 s na parę problemów; SLSQP to prototyp funkcji celu z Etapu 3). Czas pełnego LP rośnie z N w przybliżeniu kwadratowo.

## Prognoza czasu pełnego przebiegu

Krok modelowy: średnio 234 s/datę; z optymalizacją ≈ 754 s/datę.

| próba | liczba dat | jobs = 1 [h] | jobs = 4 [h] | jobs = 8 [h] | jobs = 16 [h] |
|---|---:|---:|---:|---:|---:|
| obecny plik danych | 83 | 17.4 | 4.3 | 2.2 | 1.1 |
| docelowa (kalibracja + hold-out) | 226 | 47.3 | 11.8 | 5.9 | 3.0 |

Uwagi: czas symulacji rośnie liniowo z N i (dla vine'ów) z liczbą krawędzi z kopułą t; daty wcześniejsze mają mniej aktywów. Prognoza dla próby docelowej zakłada koszt daty jak w pomiarze (≈ 80 aktywów). Analizy wrażliwości (Etap 4) i MC ex post (Etap 5) nie są wliczone.
