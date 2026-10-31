# Modele zależności na próbie kalibracyjnej

Dane: dzienne log-stopy w PLN, 2011-01-04 – 2020-12-31; 79 aktywów, 2427 sesji panelu reszt (wspólne marginesy AR(1)–GJR-GARCH(1,1) skew-t, pseudo-obserwacje z rang).
AIC = 2k − 2ℓ, BIC = k ln n − 2ℓ. Test Vuonga z poprawką Schwarza: Z > 0 – lepszy rvine; rozstrzygnięcie przy |Z| > z_(1−0.05/2). Kopuła empiryczna nie ma gęstości (bez AIC/BIC i testu).

| model | ℓ | k | AIC | BIC | Z Vuonga | p | lepszy | czas estymacji [s] |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| rvine | 160382.1 | 373 | -320018.2 | -317856.9 |  |  |  | 32.9 |
| student_t | 160323.9 | 3082 | -314483.8 | -296625.4 | 26.35 | 4.56e-153 | rvine | 1.3 |
| gauss | 160158.9 | 3081 | -314155.9 | -296303.3 | 20.88 | 8.33e-97 | rvine | 0.0 |
| dvine | 152536.6 | 374 | -304325.3 | -302158.2 | 54.45 | 0 | rvine | 32.9 |
| cvine | 103734.0 | 364 | -206740.1 | -204630.9 | 101.51 | 0 | rvine | 33.3 |
| indep | 0.0 | 0 | 0.0 | 0.0 | 128.47 | 0 | rvine | 0.0 |
| empirical |  | 0 |  |  |  |  |  | 0.0 |

## Rodziny kopuł parowych (drzewa 1–L)

### rvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton | 0 | 2 | 2 | 4 |
| clayton180 | 0 | 0 | 2 | 2 |
| frank | 0 | 13 | 20 | 33 |
| gauss | 0 | 8 | 2 | 10 |
| gumbel | 0 | 0 | 2 | 2 |
| gumbel180 | 3 | 1 | 0 | 4 |
| indep | 0 | 4 | 13 | 17 |
| t | 75 | 49 | 35 | 159 |

### cvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton180 | 0 | 1 | 0 | 1 |
| clayton270 | 0 | 1 | 0 | 1 |
| clayton90 | 0 | 1 | 0 | 1 |
| frank | 0 | 0 | 19 | 19 |
| gauss | 1 | 2 | 6 | 9 |
| gumbel | 0 | 3 | 1 | 4 |
| gumbel180 | 0 | 0 | 1 | 1 |
| gumbel270 | 0 | 7 | 0 | 7 |
| gumbel90 | 0 | 5 | 0 | 5 |
| indep | 10 | 5 | 10 | 25 |
| t | 67 | 52 | 39 | 158 |

### dvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| frank | 0 | 19 | 24 | 43 |
| gauss | 1 | 3 | 5 | 9 |
| gumbel | 0 | 1 | 1 | 2 |
| gumbel180 | 5 | 2 | 1 | 8 |
| indep | 0 | 3 | 10 | 13 |
| t | 72 | 49 | 35 | 156 |

C-vine – korzenie drzew 1–L: HYG → SCHR → EFA

D-vine – porządek zmiennych: DNP.WA – CDR.WA – KGH.WA – PKO.WA – PEO.WA – PZU.WA – PKN.WA – LPP.WA – CPS.WA – PGE.WA – TUR – EZA – EWZ – EWW – IDX – THD – INDA – MCHI – EEM – EWT – EWS – EWA – EWC – EWU – EWL – EWN – EWG – EWQ – EFA – EWJ – JPM – XOM – JNJ – PG – WMT – AMT – PLD – SPG – O – WELL – PSA – EXR – DLR – EQIX – AMZN – GOOGL – MSFT – NVDA – AAPL – FALN – ANGL – HYG – JNK – VCSH – IGSB – SHY – SCHO – VGSH – IEI – SCHR – VGIT – IEF – GOVT – IGIB – VCIT – USIG – LQD – VGLT – TLT – SGOL – GLD – IAU – FNV – WPM – GDXJ – GDX – NEM – AEM – KGC

Kopuła t: ν̂ = 24.82 (profil wiarygodności na siatce).

