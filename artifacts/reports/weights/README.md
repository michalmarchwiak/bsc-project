# Wagi tickerów w czasie

Wagi docelowe w dniu rebalansowania (pierwsza sesja miesiąca, transakcja na zamknięciu tej sesji),
przebieg `4010467a06e7`, 2008-01-02 – 2026-10-01 (226 dat). Między datami portfel jest trzymany
(wagi dryfują z cenami; obrót i koszty liczone względem wag po dryfie).

- `rvine_ce_dep_P1.csv` … `rvine_ce_dep_P5.csv` – strategia główna (R-vine × CE − kara za zależność),
  ekspozycja look-through x_i = W_k·w_{k,i} (ułamek majątku funduszu, suma w wierszu = 1);
  kolumny – tickery w kolejności sub-funduszy SF1…SF9 (tylko tickery, które choć raz miały wagę > 0).
- `artifacts/backtest/4010467a06e7/weights_tickers.parquet` – wszystkie 146 szeregów bazowych
  (7 modeli × 4 strategie × 5 profili, `ew`, `mv_hist`; model `-` = benchmark bez kopuły), format długi:
  `date, model, strategy, profile, subfund, ticker, w_sf` (udział tickera w SF, poziom 1),
  `W_sf` (waga SF, poziom 2), `x` (ekspozycja look-through).

Przykład (pandas):

```python
import pandas as pd
w = pd.read_parquet("artifacts/backtest/4010467a06e7/weights_tickers.parquet")
p3 = w[(w.model == "gauss") & (w.strategy == "ce") & (w.profile == "P3")]
p3.pivot_table(index="date", columns="ticker", values="x", observed=True).fillna(0)
```
