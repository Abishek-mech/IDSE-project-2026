"""Baselines, ML models, ARIMA wrapper, metrics, and bootstrap utilities."""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config as C

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- baselines
class MonthlyClimatology:
    """Predict the training-set mean AQI of the calendar month (no dependence on recent days)."""

    def fit(self, idx, y):
        self.m = pd.Series(np.asarray(y), index=pd.DatetimeIndex(idx).month).groupby(level=0).mean()
        return self

    def predict(self, idx):
        return pd.DatetimeIndex(idx).month.map(self.m).to_numpy(dtype=float)


# ----------------------------------------------------------------------------- ML models
def search_spaces():
    """(pipeline, hyper-parameter distributions) for each ML model."""
    ridge = Pipeline([("sc", StandardScaler()), ("m", Ridge(random_state=C.SEED))])
    rf = RandomForestRegressor(n_estimators=300, random_state=C.SEED, n_jobs=-1)
    hgb = HistGradientBoostingRegressor(random_state=C.SEED, early_stopping=False)
    return {
        "Ridge": (ridge, {"m__alpha": loguniform(1e-2, 1e3)}),
        "RandomForest": (rf, {"max_depth": randint(3, 14), "min_samples_leaf": randint(3, 40),
                              "max_features": uniform(0.3, 0.6)}),
        "GradientBoosting": (hgb, {"learning_rate": loguniform(0.01, 0.2),
                                   "max_iter": randint(80, 400), "max_leaf_nodes": randint(4, 24),
                                   "min_samples_leaf": randint(5, 60),
                                   "l2_regularization": loguniform(1e-3, 10),
                                   "loss": ["squared_error", "absolute_error"]}),
    }


def tune(name, X, y, n_iter=25, n_splits=5):
    """Randomised search with forward-chaining CV (TimeSeriesSplit) - training data only."""
    est, dist = search_spaces()[name]
    cv = TimeSeriesSplit(n_splits=n_splits)
    rs = RandomizedSearchCV(est, dist, n_iter=n_iter, cv=cv, scoring="neg_mean_absolute_error",
                            random_state=C.SEED, n_jobs=-1, refit=True)
    rs.fit(X, y)
    cv_mae = -rs.best_score_
    return rs.best_estimator_, rs.best_params_, cv_mae


class EventAdjusted:
    """Base regressor + additive festival correction.

    Rare event days (Deepavali, Bhogi) are too few for a regularised tree model to isolate. After
    fitting the base model we measure its mean residual on flagged days in the FIT data and add
    that uplift back when the flag is on. Fitted only on the data it is given (no look-ahead).
    """

    def __init__(self, est, flag_cols=("is_diwali", "is_bhogi"), min_days=5):
        self.est, self.flag_cols, self.min_days = est, list(flag_cols), min_days

    def fit(self, X, y):
        self.est.fit(X, y)
        resid = np.asarray(y, float) - self.est.predict(X)
        self.adj_ = {}
        for f in self.flag_cols:
            m = X[f].values == 1
            self.adj_[f] = float(resid[m].mean()) if m.sum() >= self.min_days else 0.0
        return self

    def predict(self, X):
        out = self.est.predict(X)
        for f, a in self.adj_.items():
            out = out + a * X[f].values
        return np.clip(out, 0, None)


# ----------------------------------------------------------------------------- ARIMA
def fourier(index, K=2):
    doy = pd.DatetimeIndex(index).dayofyear.values
    cols = {}
    for k in range(1, K + 1):
        cols[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(cols, index=pd.DatetimeIndex(index))


def select_arima_order(y_train, grid_p=(0, 1, 2, 3), grid_q=(0, 1, 2)):
    """Pick (p,0,q) by AIC on the training period only."""
    ex = fourier(y_train.index)
    best = (np.inf, None)
    rows = []
    for p in grid_p:
        for q in grid_q:
            try:
                r = SARIMAX(y_train, exog=ex, order=(p, 0, q), trend="c").fit(disp=False, maxiter=200)
                rows.append((p, q, r.aic))
                if r.aic < best[0]:
                    best = (r.aic, (p, 0, q))
            except Exception:
                continue
    return best[1], pd.DataFrame(rows, columns=["p", "q", "aic"])


def arima_one_step(series, order, fit_end, pred_start, pred_end):
    """Fit on series[:fit_end]; then roll forward WITHOUT refitting and return 1-step-ahead
    forecasts for pred_start..pred_end (each uses data up to the previous day only)."""
    tr = series.loc[:fit_end]
    res = SARIMAX(tr, exog=fourier(tr.index), order=order, trend="c").fit(disp=False, maxiter=200)
    new = series.loc[pd.Timestamp(fit_end) + pd.Timedelta(days=1):pred_end]
    ext = res.append(new, exog=fourier(new.index), refit=False)
    pm = ext.get_prediction(start=new.index[0], end=new.index[-1], dynamic=False).predicted_mean
    return pm.loc[pred_start:pred_end], res


# ----------------------------------------------------------------------------- metrics
def regression_metrics(y, p, persist=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
           "R2": r2_score(y, p), "Bias": float(np.mean(p - y))}
    if persist is not None:
        out["Skill_vs_persistence_%"] = 100 * (1 - out["MAE"] / mean_absolute_error(y, persist))
    return out


def exceedance_metrics(y, p, thr=C.AQI_THRESHOLD):
    """Detect 'Moderate or worse' days (AQI > thr) from forecast AQI."""
    yt, yp = (np.asarray(y) > thr).astype(int), (np.asarray(p) > thr).astype(int)
    return {"Precision": precision_score(yt, yp, zero_division=0),
            "Recall": recall_score(yt, yp, zero_division=0),
            "F1": f1_score(yt, yp, zero_division=0)}


def best_threshold(y, p, grid=range(70, 131, 2)):
    """Forecast-AQI cut-off that maximises F1 for the true >100 event (chosen on validation)."""
    yt = (np.asarray(y) > C.AQI_THRESHOLD).astype(int)
    scores = {t: f1_score(yt, (np.asarray(p) > t).astype(int), zero_division=0) for t in grid}
    return max(scores, key=scores.get)


def confusion(y, p, thr=C.AQI_THRESHOLD):
    return confusion_matrix((np.asarray(y) > C.AQI_THRESHOLD).astype(int),
                            (np.asarray(p) > thr).astype(int), labels=[0, 1])


def block_bootstrap_mae_diff(err_model, err_base, block=14, reps=2000, seed=C.SEED):
    """95% CI for MAE(base) - MAE(model) using a moving-block bootstrap (respects autocorrelation).
    Positive values mean the model is better than the baseline."""
    rng = np.random.default_rng(seed)
    d = np.abs(np.asarray(err_base)) - np.abs(np.asarray(err_model))
    n = len(d)
    nb = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty(reps)
    for i in range(reps):
        st = rng.integers(0, starts_max + 1, nb)
        samp = np.concatenate([d[s:s + block] for s in st])[:n]
        out[i] = samp.mean()
    return d.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)
