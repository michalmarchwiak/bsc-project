# Moduł opcji – dane i wycena (Etapy 1–2)

Moduł dostarcza dane zmienności i kwotowania opcji (offline) oraz wycenę europejskich
opcji na S&P 500 / SPY i WIG20 z parametryczną powierzchnią zmienności. Stanowi podstawę
dla sub-funduszy opcyjnych (protective put, collar) wycenianych pełnym przeszacowaniem
w scenariuszach Monte Carlo (Etapy 3–6, jeszcze niezaimplementowane).

## Uruchomienie

```bash
pip install -e ".[dev]"
python -m pipelines.options_offline          # pełny pipeline offline (Etapy 1–2)
pytest -q                                     # testy (bez sieci)
```

Poszczególne kroki: `python -m pf.data.vol_indices`, `pf.data.underlyings`,
`pf.data.options_snapshot`, `pf.data.dolthub_options [--dolt-dir KLON]`,
`pf.data.cboe_strategy_indices`, `pipelines.calibrate_surface`, `pipelines.validate_pput`.
Parametry: `config/options.yaml`. Dzienny cron: `.github/workflows/refresh.yml`
(indeksy zmienności, instrumenty bazowe, zrzut łańcuchów opcji).

## Struktura

| Plik | Zawartość |
|---|---|
| `src/pf/data/vol_indices.py` | VIX9D, VIX, VIX3M, VIX6M, VIX1Y, VVIX, SKEW, OVX, GVZ – Cboe CSV, zapasowo yfinance; walidacja |
| `src/pf/data/underlyings.py` | SPY, ^GSPC, ^SP500TR, WIG20, GLD, USO, ^IRX |
| `src/pf/data/options_snapshot.py` | dzienny zrzut łańcuchów opcji (yfinance), partycje `date=…` |
| `src/pf/data/dolthub_options.py` | historia opcji US z DoltHub `post-no-preference/options` (od 02.2019) |
| `src/pf/data/cboe_strategy_indices.py` | PPUT, PUT, BXM do walidacji |
| `src/pf/options/bs.py` | Black–Scholes–Merton z q, greki, wektoryzacja |
| `src/pf/options/iv.py` | zmienność implikowana (Newton + bisekcja), filtry kwotowań |
| `src/pf/options/surface.py` | struktura terminowa z indeksów VIX, korekta VIX→ATM, regresja skośności, OOS |
| `src/pf/options/proxy_wig20.py` | zastępcza zmienność implikowana WIG20, wrażliwość ±3 p.p. |
| `src/pf/options/pput.py` | replikacja PPUT, tracking error |
| `src/pf/artifacts.py` | zapis Parquet, `artifacts/manifest.json` (hash SHA-256, wersje pakietów, parametry) |

## Model powierzchni

Wariancja całkowita w(T) = σ²_VIX(T)·T interpolowana liniowo w T między tenorami
9/30/93/186/365 dni (płaska ekstrapolacja σ, wymuszona monotoniczność w). Dla
k = ln(K/F), F z parytetu put–call:

σ_imp(k, T) = a + b·VIX_T + c·k + d·k·VIX_T (+ e·k·SKEW/100),

gdzie a + b·VIX_T jest korektą VIX → σ_ATM. Kalibracja MNK lub Hubera (IRLS) na opcjach
OTM; walidacja na podziale czasowym (`surface.oos_split`), raport
`artifacts/reports/surface_oos.{csv,md}`. Replikacja PPUT: `artifacts/reports/pput_tracking.{csv,md}`.

## Ograniczenia

- **Powierzchnia zmienności jest odtwarzana modelowo, a nie obserwowana.** Historia
  kwotowań opcji (DoltHub) zaczyna się w 02.2019; wcześniejsze wyceny (m.in. 2008)
  są ekstrapolacją parametrów skalibrowanych na późniejszych danych.
- **VIX ≠ σ_ATM.** VIX to bezmodelowa, 30-dniowa oczekiwana wariancja z całego pasma
  cen wykonania (replikacja kontraktu wariancyjnego); przy ujemnej skośności przewyższa
  zmienność ATM. Stąd afiniczna korekta a + b·VIX_T.
- **Wycena syntetyczna zaniża ceny putów OTM bez korekty skośności.** Wycena σ = σ_ATM
  (c = d = 0) daje zbyt tanie puty ochronne (test `test_flat_vol_underprices_otm_puts_without_skew`).
- Liniowość σ w k nie oddaje wypukłości uśmiechu w dalekich skrzydłach (|k| > 0,15).
- Kwotowania, ceny bazowe i indeksy zmienności nie są idealnie zsynchronizowane w czasie
  (zamknięcie 16:00 vs 16:15 ET, zrzut yfinance po sesji).
- **WIG20 bez walidacji rynkowej.** Brak indeksu zmienności implikowanej i niska płynność
  opcji OW20; zmienność implikowana jest założeniem modelowym
  σ_imp,WIG20 = σ_GARCH,WIG20 · VIX / σ_GARCH,SPX z nachyleniem skośności z S&P 500;
  wyniki należy raportować z wrażliwością ±3 p.p.
- **Dane yfinance są nieoficjalne** (użytek osobisty/badawczy), mogą zawierać błędy
  i zmieniać format; kolumna `impliedVolatility` Yahoo jest ignorowana. Yahoo nie jest
  nigdy odpytywane z aplikacji – wyłącznie z pipeline'u offline.
- Schemat tabeli DoltHub jest weryfikowany przy każdym imporcie; przy niedostępności lub
  zmianie schematu import jest pomijany (wpis `status: unavailable` w manifeście),
  a kalibracja korzysta tylko ze zrzutów yfinance. Tabela `option_chain` nie zawiera ceny
  bazowej – dołączana jest cena zamknięcia SPY.
- Zmienność warunkowa σ_GARCH dla proxy WIG20 pochodzi z modeli marginesów, których
  repozytorium jeszcze nie zawiera – `proxy_wig20` przyjmuje ją jako wejście.
