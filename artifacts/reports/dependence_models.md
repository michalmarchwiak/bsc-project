# Ranking modeli zależności

AIC/BIC – średnie po datach (okno estymacji); Vuong – liczba dat, w których R-vine jest istotnie lepszy / gorszy (poziom 5%, poprawka Schwarza); MCS – Model Confidence Set na dziennej stracie −log-score poza próbą (α = 0.1); testy – odsetek szeregów (strategie × profile) bez odrzucenia na poziomie 0.05 w teście Kupca, Christoffersena (cc), Z₂ i Du–Escanciano; śr. ranga – średnia ranga CE portfeli modelu w każdej parze (strategia, profil) spośród 7 modeli (1 – najlepszy).

## Próba kalibracyjna

| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| gauss | -100393 | -89110 | 156 | 0 | 49.45 | nie (0.085) | 85% | 80% | 70% | 80% | 3.05 |
| rvine | -100657 | -99324 | 0 | 0 | 54.36 | tak (1.000) | 60% | 60% | 50% | 60% | 3.45 |
| student_t | -98050 | -86763 | 156 | 0 | 51.65 | nie (0.033) | 90% | 80% | 75% | 80% | 3.45 |
| cvine | -67610 | -66323 | 156 | 0 | 35.49 | nie (0.000) | 55% | 55% | 40% | 40% | 3.80 |
| dvine | -96045 | -94696 | 156 | 0 | 51.75 | nie (0.000) | 30% | 25% | 0% | 0% | 4.55 |
| indep | 0 | 0 | 156 | 0 | 0.00 | nie (0.000) | 0% | 0% | 0% | 0% | 4.80 |
| empirical |  |  | 0 | 0 |  |  () | 80% | 70% | 60% | 65% | 4.90 |

## Próba hold-out

| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cvine | -103160 | -101451 | 70 | 0 | 43.72 | nie (0.000) | 80% | 100% | 75% | 90% | 2.60 |
| empirical |  |  | 0 | 0 |  |  () | 100% | 100% | 100% | 100% | 3.35 |
| dvine | -162532 | -160819 | 70 | 0 | 71.52 | nie (0.000) | 95% | 95% | 85% | 100% | 3.50 |
| gauss | -170059 | -154502 | 70 | 0 | 75.11 | tak (1.000) | 100% | 100% | 100% | 100% | 3.55 |
| student_t | -156183 | -140621 | 70 | 0 | 71.22 | nie (0.000) | 100% | 100% | 100% | 100% | 4.90 |
| indep | 0 | 0 | 70 | 0 | 0.00 | nie (0.000) | 0% | 0% | 0% | 0% | 5.05 |
| rvine | -170094 | -168354 | 0 | 0 | 74.90 | tak (0.596) | 100% | 100% | 100% | 100% | 5.05 |

