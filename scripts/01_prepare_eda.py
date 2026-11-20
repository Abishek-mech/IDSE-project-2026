"""Step 1 - data preparation + exploratory data analysis.

Outputs: data/processed/chennai_daily_clean.csv, reports/figures/eda_*.png,
         reports/tables/data_quality_log.json, reports/tables/eda_findings.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings; warnings.filterwarnings("ignore")

from src import config as C
from src import data as D
from src.features import event_flags

sns.set_theme(style="whitegrid", context="notebook")
for p in (C.PROC_DIR, C.FIG_DIR, C.TAB_DIR):
    p.mkdir(parents=True, exist_ok=True)

df, log = D.clean()
df.to_csv(C.PROC_DIR / "chennai_daily_clean.csv")
json.dump(log, open(C.TAB_DIR / "data_quality_log.json", "w"), indent=2)
find = {}
obs = df[df["observed"]].copy()
obs["month"] = obs.index.month
obs["dow"] = obs.index.dayofweek

# 1 ------------------------------------------------------------ long-run time series
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(df.index, df["aqi"], lw=0.5, color="#8aa", label="Daily AQI")
ax.plot(df.index, df["aqi"].rolling(30, min_periods=20).mean(), lw=1.8, color="#c0392b", label="30-day mean")
for y0, lab in [(50, "Good"), (100, "Satisfactory"), (200, "Moderate")]:
    ax.axhline(y0, ls=":", lw=0.8, color="k")
    ax.text(df.index[0], y0 + 3, f"{y0}", fontsize=8)
ax.axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-31"), color="orange", alpha=0.25, label="COVID lockdown")
ax.axvline(pd.Timestamp("2021-01-01"), color="purple", ls="--", lw=1)
ax.text(pd.Timestamp("2021-01-15"), 275, "monitoring network\nexpands (~2.5 → ~7 stations)", fontsize=8, color="purple")
ax.set(title="Chennai daily AQI (CPCB bulletins), 2015-2024", ylabel="AQI")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_01_timeseries.png", dpi=150); plt.close(fig)

# 2 ------------------------------------------------------------ data-quality panel
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
m_st = df["n_stations"].resample("QE").mean()
axs[0].plot(m_st.index, m_st.values, marker="o", ms=3)
axs[0].set(title="Reporting stations (quarterly mean)", ylabel="stations")
miss = (~df["observed"]).resample("YE").sum()
axs[1].bar(miss.index.year, miss.values, color="#c0392b")
axs[1].set(title="Unusable days per year (missing/artefact)", xlabel="year")
tmp = obs.assign(chg=df["aqi"].diff().abs().reindex(obs.index))
tmp["net"] = pd.cut(tmp["n_stations"], [0, 1, 3, 6, 20], labels=["1", "2-3", "4-6", "7+"])
sns.boxplot(data=tmp, x="net", y="chg", ax=axs[2], showfliers=False, color="#9bc")
axs[2].set(title="Day-to-day |ΔAQI| by #stations", xlabel="stations reporting", ylabel="|ΔAQI|")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_02_data_quality.png", dpi=150); plt.close(fig)
find["median_abs_change_by_stations"] = tmp.groupby("net", observed=True)["chg"].median().round(1).to_dict()
find["stations_mean_by_year"] = df.groupby("year")["n_stations"].mean().round(1).to_dict()

# 3 ------------------------------------------------------------ seasonality
fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
sns.boxplot(data=obs, x="month", y="aqi", ax=axs[0], showfliers=False, color="#9bc")
axs[0].set(title="AQI by month (all years)", xlabel="month", ylabel="AQI")
dow = obs.groupby("dow")["aqi"].agg(["mean", "sem"])
axs[1].bar(range(7), dow["mean"], yerr=1.96 * dow["sem"], color="#7a9", capsize=3)
axs[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
axs[1].set(title="Mean AQI by weekday (95% CI)", ylim=(60, 100))
fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_03_seasonality.png", dpi=150); plt.close(fig)
mm = obs.groupby("month")["aqi"].mean()
find["monthly_mean_aqi"] = mm.round(1).to_dict()
find["kruskal_month_p"] = float(stats.kruskal(*[g["aqi"].values for _, g in obs.groupby("month")]).pvalue)
wk = obs[obs["dow"] >= 5]["aqi"]; wd = obs[obs["dow"] < 5]["aqi"]
find["weekend_mean"], find["weekday_mean"] = round(wk.mean(), 1), round(wd.mean(), 1)
find["weekend_vs_weekday_mannwhitney_p"] = float(stats.mannwhitneyu(wk, wd).pvalue)

# 4 ------------------------------------------------------------ yearly trend + categories
obs["cat"] = D.aqi_category(obs["aqi"])
share = pd.crosstab(obs["year"], obs["cat"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(9, 4.6))
share.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn_r", width=0.8)
ax.set(title="Share of days in each CPCB category, by year", ylabel="% of days", xlabel="year")
ax.legend(title="", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_04_categories_by_year.png", dpi=150); plt.close(fig)
find["yearly_mean_aqi"] = obs.groupby("year")["aqi"].mean().round(1).to_dict()
find["share_days_over_100_by_year"] = (obs.groupby("year")["aqi"].apply(lambda s: (s > 100).mean() * 100)).round(1).to_dict()
lock = obs.loc["2020-03-25":"2020-05-31", "aqi"].mean()
same = pd.concat([obs.loc[f"{y}-03-25":f"{y}-05-31", "aqi"] for y in (2019, 2021, 2022, 2023)]).mean()
find["covid_lockdown_mean"], find["same_window_other_years_mean"] = round(lock, 1), round(same, 1)

# 5 ------------------------------------------------------------ autocorrelation
tr_series = df.loc[:C.TRAIN_END, "aqi_filled"].dropna()
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
plot_acf(tr_series, lags=40, ax=axs[0], title="ACF (training period)")
plot_pacf(tr_series, lags=40, ax=axs[1], method="ywm", title="PACF (training period)")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_05_acf_pacf.png", dpi=150); plt.close(fig)
find["acf_lag1_7_30_365"] = [round(float(tr_series.autocorr(k)), 3) for k in (1, 7, 30, 365)]
adf = adfuller(tr_series, autolag="AIC")
find["adf_stat"], find["adf_p"] = round(float(adf[0]), 2), float(adf[1])

# 6 ------------------------------------------------------------ prominent pollutant by month
pol = df[df["observed"]].copy()
heat = pd.DataFrame({p: pol.groupby(pol.index.month)[f"top_{p}"].mean() * 100 for p in D.POLLUTANTS})
fig, ax = plt.subplots(figsize=(8, 4.6))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "% of days"})
ax.set(title="How often each pollutant is 'prominent' (drives the AQI), by month", xlabel="", ylabel="month")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_06_pollutant_by_month.png", dpi=150); plt.close(fig)
find["pollutant_share_pct_overall"] = {p: round(float(pol[f"top_{p}"].mean() * 100), 1) for p in D.POLLUTANTS}

# 7 ------------------------------------------------------------ festival effects
ev = event_flags(df.index)
rows = []
for name, col in [("Deepavali", "is_diwali"), ("Bhogi/Pongal", "is_bhogi"), ("New Year", "is_new_year")]:
    flag = ev[col].astype(bool)
    grp = (flag != flag.shift()).cumsum()
    for _, seg in df[flag].groupby(grp[flag]):
        a, b = seg.index.min(), seg.index.max()
        ctx = df.loc[a - pd.Timedelta(days=15): b + pd.Timedelta(days=15), "aqi"].copy()
        ctx = ctx[~((ctx.index >= a) & (ctx.index <= b))]
        e = seg["aqi"].mean()
        if np.isfinite(e) and ctx.notna().sum() > 10:
            rows.append({"event": name, "start": a.date(), "event_aqi": e, "context_aqi": ctx.mean()})
evdf = pd.DataFrame(rows)
evdf["uplift"] = evdf["event_aqi"] - evdf["context_aqi"]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
sns.stripplot(data=evdf, x="event", y="uplift", ax=ax, size=8, color="#c0392b")
sns.pointplot(data=evdf, x="event", y="uplift", ax=ax, color="k", linestyle="none", errorbar=None, markers="_")
ax.axhline(0, color="k", lw=0.8)
ax.set(title="AQI uplift during festival windows vs surrounding ±15 days", ylabel="Δ AQI (event − context)", xlabel="")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_07_festival_effect.png", dpi=150); plt.close(fig)
evsum = {}
for k, g in evdf.groupby("event"):
    p = stats.wilcoxon(g["uplift"]).pvalue if len(g) >= 6 else np.nan
    evsum[k] = {"n_events": int(len(g)), "mean_uplift": round(float(g["uplift"].mean()), 1), "wilcoxon_p": float(p)}
find["festival_effects"] = evsum
evdf.to_csv(C.TAB_DIR / "festival_effects.csv", index=False)

# 8 ------------------------------------------------------------ feature relevance (training period only)
from src.features import build_features
X, y, meta = build_features(df)
trm = X.index <= pd.Timestamp(C.TRAIN_END)
ok = y.notna().values & trm
corr = X[ok].apply(lambda c: stats.spearmanr(c, y[ok])[0]).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
corr.plot(kind="barh", ax=ax, color="#69b")
ax.set(title="Spearman correlation of features with next-day AQI (train only)", xlabel="ρ")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_08_feature_correlation.png", dpi=150); plt.close(fig)
find["top_feature_correlations"] = corr.abs().sort_values(ascending=False).head(6).round(3).to_dict()
find["n_modelling_rows_total"] = int(ok.sum())

json.dump(find, open(C.TAB_DIR / "eda_findings.json", "w"), indent=2, default=str)
print(json.dumps(find, indent=2, default=str))
