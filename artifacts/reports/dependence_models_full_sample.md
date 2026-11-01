# Modele zależności na próbie kalibracyjnej

Dane: dzienne log-stopy w PLN, 2008-01-02 – 2020-12-31; 62 aktywów z pokryciem ≥ 90% sesji próby, 2909 sesji panelu reszt (od debiutu najmłodszego aktywa; wspólne marginesy AR(1)–GJR-GARCH(1,1) skew-t, pseudo-obserwacje z rang).
AIC = 2k − 2ℓ, BIC = k ln n − 2ℓ. Test Vuonga z poprawką Schwarza: Z > 0 – lepszy rvine; rozstrzygnięcie przy |Z| > z_(1−0.05/2). Kopuła empiryczna nie ma gęstości (bez AIC/BIC i testu).

| model | ℓ | k | AIC | BIC | Z Vuonga | p | lepszy | czas estymacji [s] |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| student_t | 125968.1 | 1892 | -248152.2 | -236846.4 | 11.05 | 2.12e-28 | rvine | 82.3 |
| rvine | 122887.0 | 292 | -245190.0 | -243445.1 |  |  |  | 2199.4 |
| gauss | 123532.4 | 1891 | -243282.7 | -231982.9 | 12.29 | 1.03e-34 | rvine | 0.8 |
| dvine | 114675.5 | 302 | -228747.1 | -226942.4 | 56.75 | 0 | rvine | 2297.9 |
| cvine | 91142.0 | 294 | -181696.0 | -179939.2 | 81.69 | 0 | rvine | 2192.8 |
| indep | 0.0 | 0 | 0.0 | 0.0 | 111.13 | 0 | rvine | 0.0 |
| empirical |  | 0 |  |  |  |  |  | 0.0 |

## Rodziny kopuł parowych (drzewa 1–L)

### rvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton180 | 0 | 0 | 1 | 1 |
| frank | 0 | 9 | 13 | 22 |
| gauss | 0 | 3 | 3 | 6 |
| gumbel | 0 | 0 | 1 | 1 |
| gumbel180 | 2 | 1 | 3 | 6 |
| gumbel90 | 0 | 1 | 1 | 2 |
| indep | 0 | 3 | 12 | 15 |
| t | 59 | 43 | 25 | 127 |

### cvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton | 1 | 0 | 1 | 2 |
| clayton180 | 0 | 1 | 0 | 1 |
| clayton90 | 0 | 1 | 0 | 1 |
| frank | 0 | 3 | 4 | 7 |
| gauss | 0 | 2 | 2 | 4 |
| gumbel | 0 | 2 | 1 | 3 |
| gumbel180 | 3 | 0 | 5 | 8 |
| indep | 0 | 5 | 15 | 20 |
| t | 57 | 46 | 31 | 134 |

### dvine

| rodzina | T1 | T2 | T3 | razem |
|---|---:|---:|---:|---:|
| clayton180 | 0 | 0 | 1 | 1 |
| clayton270 | 0 | 0 | 1 | 1 |
| frank | 0 | 10 | 17 | 27 |
| gauss | 0 | 2 | 5 | 7 |
| gumbel | 0 | 0 | 1 | 1 |
| gumbel180 | 3 | 0 | 1 | 4 |
| gumbel90 | 0 | 0 | 1 | 1 |
| indep | 0 | 4 | 4 | 8 |
| t | 58 | 44 | 28 | 130 |

C-vine – korzenie drzew 1–L: EFA → IGIB → GDX

D-vine – porządek zmiennych: CDR.WA – LPP.WA – CPS.WA – PEO.WA – PKO.WA – KGH.WA – PKN.WA – TUR – THD – IDX – EWW – EWZ – EZA – EEM – EWT – EWS – EWA – EWC – EWU – EFA – EWQ – EWG – EWN – EWL – EWJ – XOM – JPM – NVDA – AAPL – MSFT – GOOGL – AMZN – EQIX – DLR – EXR – PSA – WELL – O – SPG – PLD – AMT – WMT – PG – JNJ – HYG – JNK – IGSB – SHY – IEI – IGIB – USIG – LQD – IEF – TLT – IAU – GLD – FNV – WPM – GDX – NEM – AEM – KGC

Kopuła t: ν̂ = 26.51 (profil wiarygodności na siatce).

