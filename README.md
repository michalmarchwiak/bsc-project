# bsc-project

Analiza ryzyka funduszu złożonego z 9 sub-funduszy (akcje PL/USA/rynki rozwinięte/wschodzące,
obligacje skarbowe USA, IG, high yield, złoto, REIT) z uwzględnieniem zależności aktywów.
Pakiet `pf` (`src/pf`): marginesy AR(1)–GJR-GARCH(1,1) ze skośnym t, modele zależności (R-, C-,
D-vine, kopuły Gaussa, t, niezależności i empiryczna), scenariusze miesięcznych stóp zwrotu w PLN,
optymalizacja składu i backtest. Waluta: PLN; stopa wolna od ryzyka: WIBOR 1M.

Dokumentacja: `docs/dependence_model.md` (model zależności), `docs/backtest.md` (backtest),
`docs/options_module.md` (moduł opcji). Parametry: `config/*.yaml`.

## Instalacja i testy

```bash
pip install -e ".[dev]"
pytest -q                    # testy bez sieci; sieciowe: pytest -m network
```

## Kolejność uruchamiania

```bash
# 1. Dane (sieć): notowania, kursy NBP, log-stopy w PLN
python data_import.py
python data_manipulation.py
python -m pf.data.riskfree                     # WIBOR 1M (zapas: stopa referencyjna NBP)

# 2. Modele zależności na próbie kalibracyjnej (raporty w artifacts/reports/)
python -m pipelines.fit_rvine_full_sample --jobs 4

# 3. Pomiar kosztu kroku modelowego i prognoza czasu backtestu
python -m pipelines.scenario_timing --jobs 3

# 4. Historyczne comiesięczne rebalansowanie składu sub-funduszy (wyniki artifacts/rebalance/)
python -m pipelines.monthly_rebalance --jobs 4
```

Moduł opcji: `pf.data.vol_indices`, `pf.data.underlyings`, `pf.data.options_snapshot`
(odświeżane codziennie przez `.github/workflows/refresh.yml`), `pipelines.calibrate_surface`,
`pipelines.validate_pput`.
