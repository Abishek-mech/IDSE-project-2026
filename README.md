# Next-day Air Quality (AQI) Forecasting for Chennai

A reproducible data-science project that forecasts Chennai's **next-day Air Quality Index** from
ten years of official CPCB daily bulletins (May 2015 – Dec 2024), compares a baseline against five
models, and analyses *where and why* the forecasts fail.

## 1. Problem statement and measurable goals

Chennai residents, schools and health agencies act on the daily AQI, but the AQI swings sharply
around festivals and seasons. **Question:** can an issuing-day model (information up to day *d-1*)
forecast the AQI of day *d* better than the obvious "tomorrow = today" rule, and can it flag
"Moderate or worse" days (AQI > 100)?

| # | Goal | Metric | Result (locked 2023-24 test set) |
|---|------|--------|-----------------------------------|
| G1 | Beat the persistence baseline by at least 5% | MAE, with 95% block-bootstrap CI of the gain excluding 0 | **8.1% lower MAE** (11.26 vs 12.26); gain 0.99 AQI pts, CI [0.45, 1.58] ✔ |
| G2 | Flag AQI > 100 days with useful precision | precision / recall / F1 | precision 0.76, recall 0.55, F1 0.64 (cut-off 90 chosen on validation); persistence F1 0.62 ✔ (modest) |
| G3 | Explain the errors, not just report them | error by AQI band, month, festival days, ablations | see `reports/` and the report ✔ |

## 2. Data

* **Source:** CPCB *Daily AQI Bulletins*, digitised by UrbanEmissions.info
  (`github.com/urbanemissionsinfo/AQI_bulletins`, licence GODL – India). 387,720 rows for all
  Indian cities; **Chennai: 3,441 daily rows**, 2015-05-01 → 2024-12-31.
* **Why this dataset:** official, real, long (3,500 days), with documented messiness (missing days,
  a changing monitoring network, outliers). Not a toy dataset.
* **Fields used:** city-average AQI, number of reporting stations, prominent pollutant(s).
* The 20 MB national file is downloaded automatically; the Chennai extract
  (`data/raw/chennai_aqi_bulletin.csv`, 175 KB) is committed so the project runs offline.

## 3. Repository layout

```
run_all.py                 # one command runs everything
src/config.py              # paths, SEED=42, split dates, event calendar
src/data.py                # download, cleaning, data-quality log
src/features.py            # leakage-safe feature engineering
src/models.py              # baselines, ML models, ARIMA, metrics, bootstrap
scripts/01_prepare_eda.py  # cleaning + 8 EDA figures + statistical tests
scripts/02_train_evaluate.py  # tuning, validation, locked test, error analysis, ablations
reports/figures/*.png      # 15 figures
reports/tables/*.csv|json  # every number quoted in the report
docs/VIVA_PREP.md          # likely viva questions with answers
```

## 4. How to run

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python run_all.py                                       # ~1-2 minutes
```

Results are deterministic (fixed seeds); two clean runs produced byte-identical result tables.

## 5. Method in brief

* **Cleaning:** reindex to a continuous daily calendar; drop 12 rows reporting AQI from 0 stations;
  remove 10 isolated single-station spikes (Hampel rule + `n_stations ≤ 1`, e.g. AQI 399 on
  2016-12-27); interpolate gaps ≤ 3 days (72 days), leave longer gaps missing (28 days);
  harmonise `OZONE`/`O3`; evaluate only on genuinely observed days.
* **Features (for target day d, all from ≤ d-1):** lags (0,1,2,3,6,13 days), 7/30-day rolling
  mean/std/max, yesterday's prominent pollutants (multi-hot), seasonal sin/cos, weekday, and
  festival flags (Deepavali, Bhogi, New Year) that are known in advance.
* **Split (chronological, never shuffled):** train 2015-05→2021 (2,233 usable days),
  validation 2022 (365), test 2023-24 (729). Hyper-parameters tuned by
  forward-chaining `TimeSeriesSplit` CV on train only; model selected on validation; final
  model refit on train+val and scored **once** on test.
* **Models:** baselines (persistence, 7-day mean, monthly climatology); ARIMA(2,0,1) with annual
  Fourier terms; Ridge; Random Forest; Gradient Boosting; Gradient Boosting + festival adjustment.

## 6. Headline results (test set, MAE in AQI points – lower is better)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Monthly climatology | 20.35 | 26.28 | -0.06 |
| 7-day mean | 15.20 | 22.61 | 0.22 |
| Persistence (baseline) | 12.26 | 18.48 | 0.48 |
| ARIMA + Fourier | 11.87 | 17.20 | 0.55 |
| Ridge | 11.91 | 16.58 | 0.58 |
| Random Forest | 11.68 | 17.05 | 0.55 |
| Gradient Boosting | 11.48 | 17.11 | 0.55 |
| **GB + festival adjustment** (selected on validation) | **11.26** | **16.05** | **0.60** |

**Key findings**
1. Chennai's AQI is highly persistent (lag-1 autocorrelation 0.74), so persistence is a hard
   baseline; the best model improves on it by ~8%, and only the tree-based models' gains are
   statistically clear.
2. Errors are concentrated in rare spikes: Deepavali and Bhogi days (MAE ≈ 49 vs 11 on normal
   days) and the Nov–Jan season. Forecasts regress to the mean – the >150 band is under-predicted
   by ~52 points on average.
3. Deepavali raises AQI by ~80 points over its surroundings (Wilcoxon p = 0.006) and Bhogi by ~33
   (p = 0.04). The effect shows up on day 0 and day +1 because bulletins cover 24 h ending in
   the afternoon.
4. The monitoring network grew from ~2.5 to ~7 stations in 2020-21; single-station days are
   noisier (median day-to-day change 17 vs 8 points). This is a structural break the models
   cannot see.

## 7. Limitations and next steps

* Only city-average AQI is available (no raw PM2.5, no per-station data) and no weather data;
  rainfall plausibly explains why some Deepavali days show no spike, but it is **not tested here**.
* Only 4 festival days fall in the test period, so event-day conclusions are indicative.
* The festival window was corrected after the first run's error analysis showed misalignment
  (decision based on training-year evidence; disclosed in the report).
* Next steps: add meteorology (rain, wind, humidity), station-level or hourly CAAQMS data,
  probabilistic (quantile) forecasts for spike risk, and test on other cities.

## 8. Reproducibility checklist

Fixed seeds ✔ · `requirements.txt` ✔ · one-command run ✔ · no random shuffling of time series ✔ ·
data cleaning logged in `reports/tables/data_quality_log.json` ✔
