# Ranking modeli zależności

AIC/BIC – średnie po datach (okno estymacji); Vuong – liczba dat, w których R-vine jest istotnie lepszy / gorszy (poziom 5%, poprawka Schwarza); MCS – Model Confidence Set na dziennej stracie −log-score poza próbą (α = 0.1); testy – odsetek szeregów (strategie × profile) bez odrzucenia na poziomie 0.05 w teście Kupca, Christoffersena (cc), Z₂ i Du–Escanciano; śr. ranga – średnia ranga CE portfeli modelu w każdej parze (strategia, profil) spośród 7 modeli (1 – najlepszy).

## Próba kalibracyjna

| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cvine | -75155 | -73942 | 156 | 0 | 35.19 | nie (0.000) | 41% | 44% | 16% | 25% | 3.06 |
| empirical |  |  | 0 | 0 |  |  () | 81% | 81% | 72% | 72% | 3.19 |
| student_t | -106101 | -93353 | 156 | 0 | 43.99 | nie (0.000) | 88% | 81% | 62% | 72% | 3.84 |
| gauss | -123222 | -110479 | 156 | 0 | 39.65 | nie (0.086) | 84% | 81% | 66% | 69% | 3.88 |
| rvine | -117396 | -116080 | 0 | 0 | 58.36 | tak (1.000) | 44% | 47% | 16% | 25% | 4.44 |
| indep | 0 | 0 | 156 | 0 | 0.00 | nie (0.000) | 0% | 0% | 3% | 0% | 4.62 |
| dvine | -111357 | -110060 | 156 | 0 | 55.62 | nie (0.000) | 0% | 0% | 0% | 0% | 4.97 |

## Próba hold-out

| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| cvine | -100537 | -98974 | 70 | 0 | 40.03 | nie (0.000) | 44% | 66% | 41% | 19% | 1.81 |
| gauss | -188542 | -172288 | 48 | 1 | 73.87 | nie (0.002) | 94% | 97% | 97% | 97% | 3.44 |
| indep | 0 | 0 | 70 | 0 | 0.00 | nie (0.000) | 0% | 0% | 0% | 0% | 3.59 |
| student_t | -124806 | -108547 | 70 | 0 | 43.11 | nie (0.000) | 88% | 97% | 88% | 97% | 4.41 |
| rvine | -179584 | -178010 | 0 | 0 | 75.99 | tak (1.000) | 78% | 97% | 78% | 84% | 4.66 |
| empirical |  |  | 0 | 0 |  |  () | 97% | 97% | 97% | 97% | 5.03 |
| dvine | -170896 | -169363 | 70 | 0 | 72.26 | nie (0.000) | 38% | 53% | 19% | 25% | 5.06 |

