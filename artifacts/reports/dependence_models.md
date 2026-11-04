# Ranking modeli zależności

AIC/BIC – średnie po datach (okno estymacji); Vuong – liczba dat, w których R-vine jest istotnie lepszy / gorszy (poziom 5%, poprawka Schwarza); MCS – Model Confidence Set na dziennej stracie −log-score poza próbą (α = 0.1); testy – odsetek szeregów (strategie × profile) bez odrzucenia na poziomie 0.05 w teście Kupca, Christoffersena (cc), Z₂ i Du–Escanciano; śr. ranga – średnia ranga CE portfeli modelu w każdej parze (strategia, profil) spośród 7 modeli (1 – najlepszy).

## Próba kalibracyjna

| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| gauss | -104480 | -92853 | 156 | 0 | 51.31 | nie (0.078) | 66% | 72% | 47% | 50% | 2.16 |
| student_t | -101453 | -89823 | 156 | 0 | 53.31 | nie (0.013) | 69% | 72% | 47% | 66% | 2.69 |
| rvine | -105615 | -104247 | 0 | 0 | 56.71 | tak (1.000) | 56% | 62% | 31% | 44% | 4.09 |
| indep | 0 | 0 | 156 | 0 | 0.00 | nie (0.000) | 0% | 0% | 6% | 0% | 4.41 |
| cvine | -71521 | -70211 | 156 | 0 | 37.04 | nie (0.000) | 44% | 44% | 28% | 16% | 4.44 |
| empirical |  |  | 0 | 0 |  |  () | 53% | 47% | 38% | 41% | 4.81 |
| dvine | -100781 | -99400 | 156 | 0 | 54.01 | nie (0.000) | 34% | 38% | 6% | 6% | 5.41 |

## Próba hold-out

| model | śr. AIC | śr. BIC | Vuong: R-vine lepszy | Vuong: R-vine gorszy | śr. log-score/dzień | MCS log-score (p) | Kupiec | Christoffersen | Z₂ | DE | śr. ranga CE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| gauss | -174912 | -158937 | 70 | 0 | 76.55 | tak (0.911) | 97% | 97% | 97% | 97% | 3.00 |
| dvine | -166880 | -165143 | 70 | 0 | 72.68 | nie (0.000) | 94% | 78% | 84% | 88% | 3.44 |
| student_t | -153008 | -137028 | 70 | 0 | 69.00 | nie (0.000) | 97% | 97% | 97% | 97% | 3.56 |
| rvine | -175550 | -173775 | 0 | 0 | 76.61 | tak (1.000) | 97% | 97% | 97% | 97% | 4.09 |
| empirical |  |  | 0 | 0 |  |  () | 97% | 97% | 97% | 97% | 4.25 |
| cvine | -105851 | -104115 | 70 | 0 | 44.54 | nie (0.000) | 78% | 97% | 78% | 78% | 4.59 |
| indep | 0 | 0 | 70 | 0 | 0.00 | nie (0.000) | 0% | 0% | 6% | 0% | 5.06 |

