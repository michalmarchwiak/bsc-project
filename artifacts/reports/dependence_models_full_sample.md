# Modele zależności na próbie kalibracyjnej

Dane: dzienne log-stopy w PLN, 2008-01-02 – 2020-12-31; 59 aktywów z pokryciem ≥ 90% sesji próby, 3287 sesji panelu reszt (od debiutu najmłodszego aktywa; wspólne marginesy AR(1)–GJR-GARCH(1,1) skew-t, pseudo-obserwacje z rang).
AIC = 2k − 2ℓ, BIC = k ln n − 2ℓ. Test Vuonga z poprawką Schwarza: Z > 0 – lepszy rvine; rozstrzygnięcie przy |Z| > z_(1−0.05/2). Kopuła empiryczna nie ma gęstości (bez AIC/BIC i testu).

| model | ℓ | k | AIC | BIC | Z Vuonga | p | lepszy | czas estymacji [s] |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| rvine | 132370.9 | 275 | -264191.7 | -262514.8 |  |  |  | 21.1 |
| gauss | 133570.0 | 1711 | -263718.1 | -253284.8 | 10.08 | 6.54e-24 | rvine | 0.0 |
| student_t | 130490.8 | 1712 | -257557.6 | -247118.2 | 14.49 | 1.39e-47 | rvine | 0.7 |
| dvine | 126641.8 | 271 | -252741.5 | -251089.0 | 41.74 | 0 | rvine | 18.6 |
| cvine | 92957.7 | 257 | -185401.3 | -183834.2 | 105.29 | 0 | rvine | 19.7 |
| indep | 0.0 | 0 | 0.0 | 0.0 | 137.70 | 0 | rvine | 0.0 |
| empirical |  | 0 |  |  |  |  |  | 0.0 |

## Rodziny kopuł parowych (drzewa 1–L)

### rvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton | 1 | 2 | 4 | 7 |
| clayton180 | 0 | 1 | 1 | 2 |
| clayton270 | 0 | 1 | 1 | 2 |
| clayton90 | 0 | 2 | 0 | 2 |
| frank | 1 | 8 | 6 | 15 |
| gauss | 0 | 4 | 6 | 10 |
| gumbel | 0 | 1 | 2 | 3 |
| gumbel180 | 8 | 4 | 0 | 12 |
| indep | 0 | 2 | 5 | 7 |
| t | 48 | 32 | 31 | 111 |

### cvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton | 2 | 0 | 4 | 6 |
| clayton270 | 0 | 0 | 1 | 1 |
| frank | 3 | 9 | 6 | 18 |
| gauss | 0 | 7 | 5 | 12 |
| gumbel | 0 | 3 | 0 | 3 |
| gumbel180 | 11 | 0 | 8 | 19 |
| gumbel270 | 0 | 0 | 2 | 2 |
| indep | 1 | 6 | 5 | 12 |
| t | 41 | 32 | 25 | 98 |

### dvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton | 1 | 2 | 1 | 4 |
| clayton180 | 0 | 0 | 1 | 1 |
| clayton270 | 0 | 0 | 1 | 1 |
| clayton90 | 0 | 0 | 1 | 1 |
| frank | 3 | 8 | 8 | 19 |
| gauss | 0 | 3 | 5 | 8 |
| gumbel180 | 7 | 4 | 0 | 11 |
| indep | 0 | 5 | 8 | 13 |
| t | 47 | 35 | 31 | 113 |

C-vine – korzenie drzew 1–L: EFA → IGIB → GDX

D-vine – porządek zmiennych: PKN.WA – PEO.WA – PKO.WA – KGH.WA – LPP.WA – CDR.WA – AMD – NVDA – NFLX – AMZN – GOOGL – AAPL – MSFT – EQIX – DLR – CCI – SBAC – AMT – PSA – WELL – VTR – SPG – PLD – BA – THD – EWW – TUR – EZA – EWZ – EEM – EWT – EWS – EWA – EWC – EWU – EFA – EWQ – EWG – EWN – EWL – EWJ – JNK – HYG – LQD – USIG – IGIB – IGSB – SHY – IEI – IEF – TLT – GLD – IAU – RGLD – WPM – GDX – AEM – NEM – AU

Kopuła t: ν̂ = 21.07 (profil wiarygodności na siatce).

