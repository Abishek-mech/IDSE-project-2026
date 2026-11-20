# Viva preparation – likely questions and honest answers

Read the code in `src/` once before the viva. The marks reward you explaining *why*, not the model's complexity.

**1. What problem did you solve and why does it matter?**
Forecast Chennai's next-day AQI from past AQI and calendar information, and flag days above 100 ("Moderate or worse"). Useful for health advisories and for planning around festival days.

**2. Why this dataset?**
Official CPCB daily bulletins, ~3,500 days, real messiness (gaps, changing network, outliers). Chennai because it has a 10-year record with few missing days.

**3. What is the baseline and why is it hard to beat?**
Persistence: tomorrow = today. Lag-1 autocorrelation is 0.74, so yesterday explains most of tomorrow. Permutation importance confirms it: shuffling `aqi_lag0` raises MAE by ~9 points, far more than any other feature.

**4. How did you avoid data leakage?**
(a) Chronological split, never shuffled. (b) Every feature for day d uses data up to d-1 only (`features.py` shifts all lag/rolling features by one day). (c) Tuning used `TimeSeriesSplit` on train only. (d) Model selection on validation; test scored once. (e) Scaling lives inside a pipeline fitted on training data. (f) Event-adjustment uplifts are computed only from the data the model is fitted on.

**5. Why MAE?** Easy to interpret in AQI points and not dominated by a few extreme days. RMSE and R² are also reported; the ranking is similar.

**6. Which model won and is the gain real?**
GB + festival adjustment: test MAE 11.26 vs 12.26 (8% better). A block bootstrap (14-day blocks, respecting autocorrelation) gives a 95% CI of 0.45–1.58 MAE points for the gain, excluding zero. ARIMA and Ridge gains are *not* significant. On validation the gains were tiny (1.7%), so be honest that the improvement is modest.

**7. Why does it fail on Deepavali/Bhogi?**
Only ~14 flagged training days, so a regularised model predicts the typical level. Hence the explicit adjustment stage. Remaining error: rainy Deepavali years show no spike (weather not in the data – a hypothesis, untested).

**8. You changed the festival window. Isn't that cheating?**
Disclose it: I noticed after the first run that test errors clustered on festival days. I then checked the offsets using *training years only* (2015-21): uplift was ~+58 on days 0 and +1 and negative on day -1, and the 2022-24 data agreed. The reason is that bulletins cover 24 h ending in the afternoon. The test set did prompt the investigation, so the test score is slightly less pristine than a first-run-only result; say so.

**9. Why did you drop `n_stations` from the features?**
The network grew from ~2.5 to ~7 stations in 2020-21. Training data never contained the new range, so trees can't extrapolate and the model would learn a regime artefact. It is documented as a limitation.

**10. Why ARIMA with Fourier terms?**
Classic statistical baseline for time series; annual seasonality handled with Fourier regressors; (p,q) chosen by AIC on train only → (2,0,1). Forecasts are one-step-ahead without refitting (parameters fixed, state updated).

**11. Why does the tuned cut-off differ from 100?**
Forecasts shrink toward the mean, so a lower cut-off (90, chosen on validation by F1) catches more true exceedances. Precision 0.76, recall 0.55 on test.

**12. What are the limitations and next steps?**
City-average AQI only, no weather, no raw PM2.5, few test festival days, structural break in monitoring. Next: rainfall/wind/humidity, hourly CAAQMS data, quantile forecasts for spike risk, other cities.

**13. Which preprocessing decisions were judgement calls?**
Dropping 12 zero-station rows; treating 10 single-station Hampel spikes as artefacts; interpolating only gaps ≤ 3 days; evaluating only on observed days. All logged in `reports/tables/data_quality_log.json`.

**14. Could you reproduce it?** Yes: `python run_all.py`, fixed seed 42; two clean runs gave byte-identical tables.
