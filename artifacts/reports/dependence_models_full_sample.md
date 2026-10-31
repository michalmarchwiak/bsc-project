# Modele zależności na próbie kalibracyjnej

Dane: dzienne log-stopy w PLN, 2008-01-02 – 2020-12-31; 79 aktywów, 3166 sesji panelu reszt (wspólne marginesy AR(1)–GJR-GARCH(1,1) skew-t, pseudo-obserwacje z rang).
AIC = 2k − 2ℓ, BIC = k ln n − 2ℓ. Test Vuonga z poprawką Schwarza: Z > 0 – lepszy rvine; rozstrzygnięcie przy |Z| > z_(1−0.05/2). Kopuła empiryczna nie ma gęstości (bez AIC/BIC i testu).

| model | ℓ | k | AIC | BIC | Z Vuonga | p | lepszy | czas estymacji [s] |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| rvine | 185285.5 | 363 | -369845.0 | -367645.1 |  |  |  | 41.5 |
| gauss | 180693.6 | 3081 | -355225.2 | -336553.7 | 24.37 | 3.72e-131 | rvine | 0.0 |
| dvine | 177035.0 | 387 | -353296.0 | -350950.7 | 40.38 | 0 | rvine | 42.0 |
| cvine | 116057.9 | 375 | -231365.8 | -229093.2 | 111.51 | 0 | rvine | 41.7 |
| indep | 0.0 | 0 | 0.0 | 0.0 | 133.65 | 0 | rvine | 0.0 |
| student_t | -593308.9 | 3082 | 1192781.8 | 1211459.4 | 335.49 | 0 | rvine | 1.6 |
| empirical |  | 0 |  |  |  |  |  | 0.0 |

## Rodziny kopuł parowych (drzewa 1–L)

### rvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton180 | 0 | 0 | 1 | 1 |
| frank | 0 | 15 | 16 | 31 |
| gauss | 0 | 5 | 3 | 8 |
| gumbel | 0 | 0 | 1 | 1 |
| gumbel180 | 1 | 1 | 4 | 6 |
| gumbel270 | 0 | 0 | 1 | 1 |
| gumbel90 | 0 | 1 | 0 | 1 |
| indep | 0 | 5 | 20 | 25 |
| t | 77 | 50 | 30 | 157 |

### cvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton180 | 0 | 0 | 1 | 1 |
| frank | 0 | 1 | 5 | 6 |
| gauss | 2 | 6 | 15 | 23 |
| gumbel | 1 | 0 | 4 | 5 |
| gumbel180 | 0 | 2 | 0 | 2 |
| gumbel270 | 0 | 8 | 0 | 8 |
| gumbel90 | 0 | 0 | 2 | 2 |
| indep | 9 | 5 | 6 | 20 |
| t | 66 | 55 | 43 | 164 |

### dvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton180 | 0 | 0 | 1 | 1 |
| clayton270 | 0 | 0 | 1 | 1 |
| frank | 1 | 17 | 22 | 40 |
| gauss | 2 | 6 | 8 | 16 |
| gumbel180 | 2 | 3 | 0 | 5 |
| gumbel90 | 0 | 0 | 2 | 2 |
| indep | 0 | 1 | 4 | 5 |
| t | 73 | 50 | 38 | 161 |

C-vine – korzenie drzew 1–L: HYG → EFA → IEI

D-vine – porządek zmiennych: DNP.WA – LPP.WA – CPS.WA – CDR.WA – PKN.WA – KGH.WA – PKO.WA – PEO.WA – PZU.WA – PGE.WA – INDA – MCHI – IDX – THD – TUR – EWW – EWZ – EZA – EEM – EWT – EWS – EWA – EWC – EWU – EFA – EWQ – EWG – EWN – EWL – EWJ – XOM – JPM – MSFT – AMZN – GOOGL – AAPL – NVDA – EQIX – DLR – EXR – PSA – WELL – O – SPG – PLD – AMT – WMT – PG – JNJ – HYG – JNK – FALN – ANGL – GOVT – VCIT – VGIT – SCHR – SCHO – VGSH – VCSH – SHY – IGSB – IGIB – LQD – USIG – IEI – IEF – TLT – VGLT – IAU – GLD – SGOL – GDXJ – WPM – GDX – NEM – AEM – KGC – FNV

Kopuła t: ν̂ = 2.94 (profil wiarygodności na siatce).

