# Diagnostyka zależności w horyzoncie miesiąca

Daty: 19 (2008-01-02 – 2026-01-02), uniwersum: pit, H = 21 sesji, N = 5000 scenariuszy, q = 0.1. Porównanie stóp H-dniowych portfeli równych wag sub-funduszy: dane okna estymacji a scenariusze modelu z kopułą szacowaną na sumach reszt z k sesji (k = 1: panel dzienny). ρ_S – korelacja rang, Λ(q) – zależność w dolnym ogonie; błąd = scenariusze − dane, średnie po parach SF i datach.

| model | k | pary | ρ_S dane | ρ_S scen. | średni błąd ρ_S | MAE ρ_S | Λ dane | Λ scen. | średni błąd Λ | MAE Λ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gauss | 1 | GPW × USA | 0.027 | 0.033 | +0.007 | 0.149 | 0.236 | 0.118 | -0.118 | 0.129 |
| gauss | 2 | GPW × USA | 0.027 | 0.037 | +0.011 | 0.120 | 0.236 | 0.124 | -0.112 | 0.120 |
| gauss | 5 | GPW × USA | 0.027 | 0.027 | +0.000 | 0.091 | 0.236 | 0.127 | -0.110 | 0.116 |
| gauss | 10 | GPW × USA | 0.027 | 0.009 | -0.018 | 0.088 | 0.236 | 0.122 | -0.114 | 0.119 |
| gauss | 1 | ten sam rynek | 0.394 | 0.457 | +0.063 | 0.119 | 0.308 | 0.333 | +0.025 | 0.083 |
| gauss | 2 | ten sam rynek | 0.394 | 0.405 | +0.012 | 0.087 | 0.308 | 0.303 | -0.005 | 0.073 |
| gauss | 5 | ten sam rynek | 0.394 | 0.372 | -0.021 | 0.072 | 0.308 | 0.285 | -0.024 | 0.074 |
| gauss | 10 | ten sam rynek | 0.394 | 0.364 | -0.030 | 0.066 | 0.308 | 0.281 | -0.027 | 0.075 |
| gauss | 1 | wszystkie pary | 0.314 | 0.365 | +0.051 | 0.125 | 0.292 | 0.286 | -0.006 | 0.093 |
| gauss | 2 | wszystkie pary | 0.314 | 0.325 | +0.012 | 0.094 | 0.292 | 0.264 | -0.029 | 0.084 |
| gauss | 5 | wszystkie pary | 0.314 | 0.297 | -0.017 | 0.076 | 0.292 | 0.250 | -0.042 | 0.083 |
| gauss | 10 | wszystkie pary | 0.314 | 0.287 | -0.027 | 0.070 | 0.292 | 0.247 | -0.046 | 0.084 |

Pojedyncza korelacja z danych ma błąd standardowy rzędu 0,1 (ok. 60 rozłącznych okresów w oknie), więc MAE nie spada do zera nawet dla poprawnego modelu; miarodajne jest porównanie wariantów k.
