"""Step 2 - modelling, validation, locked test evaluation, error analysis, ablations.

Protocol (no leakage):
  * Chronological split: train 2015-05..2021, validation 2022, test 2023-2024 (touched once).
  * Hyper-parameters tuned with forward-chaining TimeSeriesSplit CV on TRAIN only.
  * Model selection on VALIDATION. Final models refit on train+val, scored once on TEST.
  * Features for target day d use information up to d-1 only (see src/features.py).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src import config as C
from src import data as D
from src import features as F
from src import models as M

np.random.seed(C.SEED)
sns.set_theme(style="whitegrid", context="notebook")
for p in (C.FIG_DIR, C.TAB_DIR, C.MODEL_DIR, C.PROC_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ data & split
df, _ = D.clean(verbose=False)
X, y, meta = F.build_features(df)
tr, va, te = F.split_masks(X.index)
has_y = y.notna().values
ev_tr, ev_va, ev_te = tr & has_y, va & has_y, te & has_y         # rows usable for fit / scoring
dev = (tr | va) & has_y
print(f"rows  train={ev_tr.sum()}  val={ev_va.sum()}  test={ev_te.sum()}")
LIN_COLS = [c for c in X.columns if c != "dow"]                    # raw weekday code is meaningless for a linear model

preds_val, preds_test, info = {}, {}, {}

# ------------------------------------------------------------------ baselines
preds_val["Persistence"] = meta["persistence"].values
preds_test["Persistence"] = meta["persistence"].values
preds_val["7-day mean"] = X["roll_mean7"].values
preds_test["7-day mean"] = X["roll_mean7"].values
clim_tr = M.MonthlyClimatology().fit(X.index[ev_tr], y[ev_tr])
clim_dev = M.MonthlyClimatology().fit(X.index[dev], y[dev])
preds_val["Monthly climatology"] = clim_tr.predict(X.index)
preds_test["Monthly climatology"] = clim_dev.predict(X.index)

# ------------------------------------------------------------------ ARIMA (+ annual Fourier terms)
ser = df["aqi_filled"]
order, aic_tab = M.select_arima_order(ser.loc[:C.TRAIN_END])
aic_tab.to_csv(C.TAB_DIR / "arima_order_search.csv", index=False)
print("ARIMA order (AIC, train only):", order)
v0, v1 = X.index[va][0], X.index[va][-1]
t0, t1 = X.index[te][0], X.index[te][-1]
pa_val, _ = M.arima_one_step(ser, order, C.TRAIN_END, v0, v1)
pa_test, _ = M.arima_one_step(ser, order, C.VAL_END, t0, t1)
preds_val["ARIMA+Fourier"] = pa_val.reindex(X.index).values
preds_test["ARIMA+Fourier"] = pa_test.reindex(X.index).values
info["ARIMA+Fourier"] = {"order": list(order)}

# ------------------------------------------------------------------ ML models: tune -> validate -> refit -> test
fitted_dev = {}
for name in ["Ridge", "RandomForest", "GradientBoosting"]:
    cols = LIN_COLS if name == "Ridge" else list(X.columns)
    est, best, cv_mae = M.tune(name, X.loc[ev_tr, cols], y[ev_tr])
    preds_val[name] = pd.Series(est.predict(X[cols]), index=X.index).values
    final = clone(est).fit(X.loc[dev, cols], y[dev])                # refit on train+val
    preds_test[name] = pd.Series(final.predict(X[cols]), index=X.index).values
    fitted_dev[name] = (final, cols)
    if name == "GradientBoosting":                                   # hybrid: base GB + festival uplift
        h_tr = M.EventAdjusted(clone(est)).fit(X.loc[ev_tr, cols], y[ev_tr])
        h_dev = M.EventAdjusted(clone(est)).fit(X.loc[dev, cols], y[dev])
        preds_val["GB + event adjustment"] = pd.Series(h_tr.predict(X[cols]), index=X.index).values
        preds_test["GB + event adjustment"] = pd.Series(h_dev.predict(X[cols]), index=X.index).values
        info["GB + event adjustment"] = {"uplift_train_only": h_tr.adj_, "uplift_train_val": h_dev.adj_}
        print("event uplifts (train, train+val):", h_tr.adj_, h_dev.adj_)
    info[name] = {"best_params": {k: (float(v) if hasattr(v, "item") else v) for k, v in best.items()},
                  "cv_mae_train": float(cv_mae)}
    print(name, f"CV-MAE={cv_mae:.2f}", best)

# ------------------------------------------------------------------ scoring helpers
persist = meta["persistence"].values


def table(preds, mask):
    rows = []
    for n, p in preds.items():
        p = np.asarray(p, float)
        m = mask & np.isfinite(p)
        r = M.regression_metrics(y[m], p[m], persist[m])
        r.update(M.exceedance_metrics(y[m], p[m]))
        r["n"] = int(m.sum())
        rows.append({"Model": n, **r})
    return pd.DataFrame(rows).set_index("Model")


val_tab, test_tab = table(preds_val, ev_va), table(preds_test, ev_te)
val_tab.round(3).to_csv(C.TAB_DIR / "results_validation.csv")
test_tab.round(3).to_csv(C.TAB_DIR / "results_test.csv")
print("\nVALIDATION\n", val_tab.round(2), "\n\nTEST\n", test_tab.round(2))

ml_models = ["ARIMA+Fourier", "Ridge", "RandomForest", "GradientBoosting", "GB + event adjustment"]
best = val_tab.loc[ml_models, "MAE"].idxmin()                       # selection on VALIDATION only
print("\nSelected on validation:", best)
info["selected_model"] = best
bp_val, bp_test = np.asarray(preds_val[best], float), np.asarray(preds_test[best], float)

# bootstrap CI of improvement over persistence on TEST (all models)
boot = []
for n in test_tab.index:
    if n == "Persistence":
        continue
    m = ev_te & np.isfinite(np.asarray(preds_test[n], float))
    e_m = np.asarray(preds_test[n], float)[m] - y[m].values
    e_b = persist[m] - y[m].values
    d, lo, hi = M.block_bootstrap_mae_diff(e_m, e_b)
    boot.append({"Model": n, "MAE_gain_vs_persistence": d, "CI95_low": lo, "CI95_high": hi})
boot = pd.DataFrame(boot).set_index("Model")
boot.round(2).to_csv(C.TAB_DIR / "bootstrap_vs_persistence_test.csv")
print("\nBootstrap (MAE gain vs persistence, test)\n", boot.round(2))

# threshold for 'Moderate or worse' chosen on validation, applied unchanged on test
thr = M.best_threshold(y[ev_va], bp_val[ev_va])
info["tuned_threshold"] = int(thr)
exc = {"default_100": M.exceedance_metrics(y[ev_te], bp_test[ev_te]),
       f"tuned_{thr}": M.exceedance_metrics(y[ev_te], bp_test[ev_te], thr),
       "persistence_100": M.exceedance_metrics(y[ev_te], persist[ev_te])}
info["exceedance_test"] = exc
print("\nExceedance (AQI>100) on test:", json.dumps(exc, indent=1))

# ------------------------------------------------------------------ error analysis (test)
res = pd.DataFrame({"actual": y[ev_te], "pred": bp_test[ev_te], "persist": persist[ev_te]}, index=X.index[ev_te])
res["err"] = res["pred"] - res["actual"]
res["abs_err"] = res["err"].abs()
res["abs_err_persist"] = (res["persist"] - res["actual"]).abs()
res["band"] = pd.cut(res["actual"], [0, 50, 100, 150, 400], labels=["≤50", "51-100", "101-150", ">150"])
res["month"] = res.index.month
res["is_diwali"] = X.loc[res.index, "is_diwali"].values
res.to_csv(C.PROC_DIR / "test_predictions.csv")
by_band = res.groupby("band", observed=True).agg(n=("err", "size"), MAE=("abs_err", "mean"),
                                                   MAE_persist=("abs_err_persist", "mean"), Bias=("err", "mean"))
by_month = res.groupby("month").agg(n=("err", "size"), MAE=("abs_err", "mean"),
                                    MAE_persist=("abs_err_persist", "mean"), Bias=("err", "mean"))
diw = res.groupby("is_diwali").agg(n=("err", "size"), MAE=("abs_err", "mean"), Bias=("err", "mean"))
worst = res.sort_values("abs_err", ascending=False).head(10)[["actual", "pred", "persist", "err"]].round(1)
by_band.round(2).to_csv(C.TAB_DIR / "error_by_band.csv")
by_month.round(2).to_csv(C.TAB_DIR / "error_by_month.csv")
diw.round(2).to_csv(C.TAB_DIR / "error_diwali.csv")
worst.to_csv(C.TAB_DIR / "worst_10_days_test.csv")
print("\nBy band\n", by_band.round(1), "\n\nDiwali\n", diw.round(1), "\n\nWorst\n", worst)

# ------------------------------------------------------------------ permutation importance (explanation only)
base_name = "GradientBoosting" if best == "GB + event adjustment" else best   # explain the underlying ML model
if base_name in fitted_dev:
    mdl, cols = fitted_dev[base_name]
    pi = permutation_importance(mdl, X.loc[ev_te, cols], y[ev_te], n_repeats=20, random_state=C.SEED,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    imp.round(3).to_csv(C.TAB_DIR / "permutation_importance_test.csv")
    joblib.dump(mdl, C.MODEL_DIR / "best_model.joblib")
else:
    imp = None

# ------------------------------------------------------------------ ablations (final tuned config, test MAE)
abl = {}
if base_name in fitted_dev:
    base_est, base_cols = fitted_dev[base_name]
    groups = {"all features": [],
              "no pollutant flags": [c for c in X.columns if c.startswith("top_")],
              "no festival flags": ["is_diwali", "is_bhogi", "is_new_year"],
              "no calendar/season": ["doy_sin", "doy_cos", "dow", "is_weekend"],
              "no rolling stats": ["roll_mean7", "roll_mean30", "roll_std7", "roll_max7", "delta_vs_mean7"]}
    for g, drop in groups.items():
        cols = [c for c in base_cols if c not in drop]
        m = clone(base_est).fit(X.loc[dev, cols], y[dev])
        p = m.predict(X.loc[ev_te, cols])
        abl[g] = float(M.regression_metrics(y[ev_te], p)["MAE"])
    if best == "GB + event adjustment":                                 # value of the festival stage itself
        abl["no event adjustment (base GB)"] = float(M.regression_metrics(
            y[ev_te], np.asarray(preds_test["GradientBoosting"], float)[ev_te])["MAE"])
    # training-window ablation: only post-2020 data (new monitoring network / post-COVID)
    recent = dev & (X.index >= pd.Timestamp("2020-06-01"))
    m = clone(base_est).fit(X.loc[recent, base_cols], y[recent])
    abl["train only Jun-2020 onward"] = float(M.regression_metrics(y[ev_te], m.predict(X.loc[ev_te, base_cols]))["MAE"])
    pd.Series(abl, name="test_MAE").round(3).to_csv(C.TAB_DIR / "ablation_test_mae.csv")
    print("\nAblation (test MAE):", {k: round(v, 2) for k, v in abl.items()})
info["ablation_test_mae"] = abl
json.dump(info, open(C.TAB_DIR / "model_info.json", "w"), indent=2, default=str)

# ================================================================== FIGURES
order_names = list(test_tab.index)
fig, ax = plt.subplots(figsize=(10, 4.4))
w = 0.38
xs = np.arange(len(order_names))
ax.bar(xs - w / 2, val_tab.loc[order_names, "MAE"], w, label="Validation (2022)", color="#7aa6c2")
ax.bar(xs + w / 2, test_tab.loc[order_names, "MAE"], w, label="Test (2023-24)", color="#c0392b")
for i, n in enumerate(order_names):
    ax.text(i + w / 2, test_tab.loc[n, "MAE"] + 0.2, f"{test_tab.loc[n, 'MAE']:.1f}", ha="center", fontsize=8)
ax.set_xticks(xs, order_names, rotation=20, ha="right")
ax.set(ylabel="MAE (AQI points)", title="Next-day AQI forecast error by model (lower is better)")
ax.legend(); fig.tight_layout(); fig.savefig(C.FIG_DIR / "res_01_model_comparison.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(2, 1, figsize=(13, 7.5))
axs[0].plot(res.index, res["actual"], color="k", lw=0.9, label="Actual")
axs[0].plot(res.index, res["pred"], color="#c0392b", lw=0.9, label=f"{best}")
axs[0].plot(res.index, res["persist"], color="#7aa6c2", lw=0.6, alpha=0.7, label="Persistence")
axs[0].axhline(C.AQI_THRESHOLD, ls=":", color="k", lw=0.8)
axs[0].set(title="Test period: next-day AQI forecasts", ylabel="AQI"); axs[0].legend(ncol=3)
z = res.loc["2023-10-01":"2024-02-15"]
axs[1].plot(z.index, z["actual"], color="k", marker="o", ms=2.5, lw=0.9, label="Actual")
axs[1].plot(z.index, z["pred"], color="#c0392b", marker="o", ms=2.5, lw=0.9, label=f"{best}")
axs[1].axhline(C.AQI_THRESHOLD, ls=":", color="k", lw=0.8)
axs[1].set(title="Zoom: Oct 2023 – Feb 2024 (Deepavali, Bhogi)", ylabel="AQI"); axs[1].legend()
fig.tight_layout(); fig.savefig(C.FIG_DIR / "res_02_test_forecasts.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
bb = by_band.reset_index()
xb = np.arange(len(bb))
axs[0].bar(xb - 0.2, bb["MAE"], 0.4, label=best, color="#c0392b")
axs[0].bar(xb + 0.2, bb["MAE_persist"], 0.4, label="Persistence", color="#7aa6c2")
axs[0].set_xticks(xb, bb["band"].astype(str)); axs[0].set(title="MAE by true AQI band", xlabel="actual AQI", ylabel="MAE"); axs[0].legend()
ax2 = axs[1]
ax2.bar(by_band.index.astype(str), by_band["Bias"], color="#69b")
ax2.axhline(0, color="k", lw=0.8)
ax2.set(title="Mean error (pred − actual) by band\n→ regression to the mean at extremes", xlabel="actual AQI")
axs[2].scatter(res["actual"], res["pred"], s=8, alpha=0.5, color="#c0392b")
lim = [res[["actual", "pred"]].min().min() - 5, res[["actual", "pred"]].max().max() + 5]
axs[2].plot(lim, lim, "k--", lw=0.8)
axs[2].set(title="Predicted vs actual (test)", xlabel="actual", ylabel="predicted")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "res_03_error_analysis.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(by_month.index.astype(str), by_month["MAE"], color="#c0392b", label=best, width=0.4, align="edge")
ax.bar(by_month.index.astype(str), by_month["MAE_persist"], color="#7aa6c2", label="Persistence", width=-0.4, align="edge")
ax.set(title="Test MAE by month", xlabel="month", ylabel="MAE"); ax.legend()
fig.tight_layout(); fig.savefig(C.FIG_DIR / "res_04_error_by_month.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, (lab, p, t) in zip(axs, [(f"{best} (cut-off {thr})", bp_test, thr), ("Persistence (cut-off 100)", persist, 100)]):
    cm = M.confusion(y[ev_te], p[ev_te], t)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["≤100", ">100"], yticklabels=["≤100", ">100"])
    ax.set(title=lab, xlabel="predicted", ylabel="actual")
fig.suptitle("Detecting 'Moderate or worse' days (test)"); fig.tight_layout()
fig.savefig(C.FIG_DIR / "res_05_confusion.png", dpi=150); plt.close(fig)

if imp is not None:
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.head(15)[::-1].plot(kind="barh", ax=ax, color="#69b")
    ax.set(title=f"Permutation importance - {base_name} (test)", xlabel="increase in MAE when shuffled")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "res_06_importance.png", dpi=150); plt.close(fig)

if abl:
    fig, ax = plt.subplots(figsize=(8, 4))
    a = pd.Series(abl)
    a.plot(kind="barh", ax=ax, color=["#c0392b" if k == "all features" else "#9bc" for k in a.index])
    ax.set(title="Ablation: test MAE when removing feature groups", xlabel="MAE")
    ax.set_xlim(left=max(0, a.min() - 2))
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "res_07_ablation.png", dpi=150); plt.close(fig)

print("\nDone.")
