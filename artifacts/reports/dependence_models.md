# Ranking modeli zależności

AIC/BIC – średnie po datach (okno estymacji); Vuong – liczba dat, w których R-vine jest istotnie lepszy / gorszy (poziom 5%, poprawka Schwarza); MCS – Model Confidence Set na dziennej stracie −log-score poza próbą (α = 0.1); testy – odsetek szeregów (strategie × profile) bez odrzucenia na poziomie 0.05 w teście Kupca, Christoffersena (cc), Z₂ i Du–Escanciano; śr. ranga – średnia ranga CE portfeli modelu w każdej parze (strategia, profil) spośród 7 modeli (1 – najlepszy).

## Próba kalibracyjna

| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cvine | -75155 | -73942 | 156 | 0 | 35.19 | nie (0.000) | 25% | 25% | 6% | 6% | 2.84 |
| empirical |  |  | 0 | 0 |  |  () | 81% | 81% | 56% | 72% | 3.12 |
| rvine | -117396 | -116080 | 0 | 0 | 58.36 | tak (1.000) | 34% | 38% | 16% | 19% | 3.69 |
| student_t | -106101 | -93353 | 156 | 0 | 43.99 | nie (0.000) | 75% | 75% | 50% | 62% | 4.38 |
| gauss | -123222 | -110479 | 156 | 0 | 39.65 | nie (0.086) | 75% | 75% | 50% | 62% | 4.50 |
| dvine | -111357 | -110060 | 156 | 0 | 55.62 | nie (0.000) | 0% | 0% | 0% | 0% | 4.59 |
| indep | 0 | 0 | 156 | 0 | 0.00 | nie (0.000) | 0% | 0% | 9% | 0% | 4.88 |

## Próba hold-out

| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cvine | -100537 | -98974 | 70 | 0 | 40.03 | nie (0.000) | 50% | 75% | 38% | 12% | 2.16 |
| gauss | -188542 | -172288 | 48 | 1 | 73.87 | nie (0.002) | 91% | 97% | 91% | 91% | 3.00 |
| indep | 0 | 0 | 70 | 0 | 0.00 | nie (0.000) | 0% | 0% | 3% | 0% | 4.09 |
| student_t | -124806 | -108547 | 70 | 0 | 43.11 | nie (0.000) | 78% | 97% | 78% | 94% | 4.38 |
| rvine | -179584 | -178010 | 0 | 0 | 75.99 | tak (1.000) | 78% | 97% | 78% | 78% | 4.56 |
| empirical |  |  | 0 | 0 |  |  () | 97% | 97% | 97% | 97% | 4.78 |
| dvine | -170896 | -169363 | 70 | 0 | 72.26 | nie (0.000) | 34% | 53% | 19% | 25% | 5.03 |

