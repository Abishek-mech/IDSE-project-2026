"""Feature engineering for next-day AQI forecasting.

Row indexed by TARGET day d. Every lag/rolling feature uses information up to day d-1 only
(the forecast is issued on the evening of d-1). Calendar features of day d are legitimately
known in advance. No future information enters any feature -> no leakage.
"""
import numpy as np
import pandas as pd

from . import config as C
from .data import POLLUTANTS

LAGS = [0, 1, 2, 3, 6, 13]      # 'aqi_lagk' = AQI k days before the forecast issue day (d-1)


def event_flags(index):
    """Fireworks / festival windows that plausibly spike pollution in Chennai."""
    idx = pd.DatetimeIndex(index)
    diw = pd.DatetimeIndex(pd.to_datetime(C.DEEPAVALI))
    diwali = np.zeros(len(idx), dtype=int)
    for d in diw:
        # Window = Deepavali day and the day after. Chosen from TRAINING years (2015-2021): mean AQI uplift
        # was ~+58 on days 0/+1 but negative on day -1. The CPCB bulletin covers 24 h ending mid-afternoon,
        # so the evening fireworks appear in the next day's reading.
        diwali |= ((idx >= d) & (idx <= d + pd.Timedelta(days=1))).astype(int)
    bhogi = ((idx.month == 1) & (idx.day.isin([13, 14]))).astype(int)          # Bhogi bonfires
    nye = (((idx.month == 12) & (idx.day == 31)) | ((idx.month == 1) & (idx.day == 1))).astype(int)
    return pd.DataFrame({"is_diwali": diwali, "is_bhogi": bhogi, "is_new_year": nye}, index=idx)


def build_features(df):
    """df: output of data.clean(). Returns (X, y, meta) aligned on target date."""
    s = df["aqi_filled"]                      # imputed (<=3-day gaps) series for lag construction
    # Features known on issue day d-1 ...
    base = pd.DataFrame(index=df.index)
    for k in LAGS:
        base[f"aqi_lag{k}"] = s.shift(k)
    base["roll_mean7"] = s.rolling(7, min_periods=6).mean()
    base["roll_mean30"] = s.rolling(30, min_periods=25).mean()
    base["roll_std7"] = s.rolling(7, min_periods=6).std()
    base["roll_max7"] = s.rolling(7, min_periods=6).max()
    base["delta_vs_mean7"] = s - base["roll_mean7"]
    for p in POLLUTANTS:
        base[f"top_{p}"] = df[f"top_{p}"]
    base["top_PM_share7"] = df["top_PM2.5"].rolling(7, min_periods=1).mean()

    # ... shifted forward by one day so they are attached to the target day d
    feats = base.shift(1)
    # Rename: after shift, 'aqi_lag0' on row d is the AQI of d-1 (latest known value)
    d = df.index
    cal = pd.DataFrame(index=d)
    doy = d.dayofyear.values
    cal["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    cal["dow"] = d.dayofweek
    cal["is_weekend"] = (d.dayofweek >= 5).astype(int)
    ev = event_flags(d)
    X = pd.concat([feats, cal, ev], axis=1)
    y = df["aqi"]                              # target: genuine observed AQI (NaN if missing)
    meta = pd.DataFrame({"observed": df["observed"], "persistence": feats["aqi_lag0"]})
    keep = X.notna().all(axis=1)
    return X[keep], y[keep], meta[keep]


def split_masks(index):
    idx = pd.DatetimeIndex(index)
    tr = idx <= pd.Timestamp(C.TRAIN_END)
    va = (idx > pd.Timestamp(C.TRAIN_END)) & (idx <= pd.Timestamp(C.VAL_END))
    te = idx > pd.Timestamp(C.VAL_END)
    return tr, va, te
