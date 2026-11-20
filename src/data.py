"""Loading, cleaning and quality-logging of the CPCB daily AQI bulletin data for Chennai."""
import json
import urllib.request

import pandas as pd

from . import config as C

POLLUTANTS = ["PM2.5", "PM10", "CO", "O3", "NO2", "SO2"]


def download_raw(force=False):
    """Download the national CPCB bulletin master file (~20 MB) and cache it."""
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not C.RAW_FILE.exists():
        print(f"Downloading {C.RAW_URL}")
        urllib.request.urlretrieve(C.RAW_URL, C.RAW_FILE)
    return C.RAW_FILE


def make_city_extract():
    """Slice the city out of the national file and save a small CSV used by the pipeline."""
    if C.CITY_FILE.exists():
        return C.CITY_FILE
    download_raw()
    df = pd.read_csv(C.RAW_FILE, index_col=0, parse_dates=["date"])
    df = df[df["City"] == C.CITY].sort_values("date")   # exact match: excludes 'Chengalpattu'
    df.to_csv(C.CITY_FILE, index=False)
    return C.CITY_FILE


def load_raw_city():
    make_city_extract()
    return pd.read_csv(C.CITY_FILE, parse_dates=["date"])


def _pollutant_flags(s):
    """'CO, OZONE, PM2.5' -> multi-hot columns; harmonises OZONE/O3 naming."""
    s = s.fillna("").astype(str).str.replace("OZONE", "O3", regex=False)
    return pd.DataFrame({f"top_{p}": s.str.contains(p, regex=False).astype(int) for p in POLLUTANTS},
                        index=s.index)


def clean(raw=None, verbose=True):
    """Return a continuous daily DataFrame and a data-quality log (dict)."""
    raw = load_raw_city() if raw is None else raw.copy()
    log = {"raw_rows": int(len(raw))}
    raw = raw.rename(columns={"No. Stations": "n_stations", "Air Quality": "category",
                              "Index Value": "aqi_raw", "Prominent Pollutant": "pollutants"})

    log["duplicate_dates_removed"] = int(raw["date"].duplicated().sum())
    raw = raw.drop_duplicates("date").set_index("date").sort_index()

    # Data-type handling
    raw["aqi_raw"] = pd.to_numeric(raw["aqi_raw"], errors="coerce")
    raw["n_stations"] = pd.to_numeric(raw["n_stations"], errors="coerce")

    # Rows claiming an AQI from 0 reporting stations (May 2015 start-up) are not trustworthy
    zero = raw["n_stations"] == 0
    log["zero_station_rows_dropped"] = int(zero.sum())
    raw = raw[~zero]

    # Reindex to a full daily calendar so gaps become explicit
    full = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
    df = raw.reindex(full)
    df.index.name = "date"
    log["calendar_days"] = int(len(full))
    log["missing_days_before_imputation"] = int(df["aqi_raw"].isna().sum())

    # Outliers: isolated single-station spikes (Hampel rule) are treated as artefacts.
    med = df["aqi_raw"].rolling(15, center=True, min_periods=5).median()
    mad = (df["aqi_raw"] - med).abs().rolling(15, center=True, min_periods=5).median() * 1.4826
    hampel = (df["aqi_raw"] - med).abs() > 4 * mad.clip(lower=8)
    artefact = hampel & (df["n_stations"] <= 1)
    log["hampel_flagged_total"] = int(hampel.sum())
    log["single_station_artefacts_removed"] = int(artefact.sum())
    log["artefact_dates"] = [str(d.date()) for d in df.index[artefact]]
    df["aqi"] = df["aqi_raw"].mask(artefact)

    # Short gaps -> linear interpolation, long gaps stay NaN (not invented)
    n_before = int(df["aqi"].isna().sum())
    df["aqi_filled"] = df["aqi"].interpolate(limit=C.MAX_INTERP_GAP, limit_area="inside")
    log["days_interpolated"] = int(n_before - df["aqi_filled"].isna().sum())
    log["days_left_missing"] = int(df["aqi_filled"].isna().sum())
    df["observed"] = df["aqi"].notna()          # evaluation is restricted to truly observed days

    flags = _pollutant_flags(df["pollutants"])
    df = pd.concat([df, flags], axis=1)
    df["n_stations"] = df["n_stations"].ffill().bfill()
    df["year"] = df.index.year
    if verbose:
        print(json.dumps(log, indent=2))
    return df, log


def aqi_category(x):
    return pd.cut(x, C.CPCB_BINS, labels=C.CPCB_LABELS, right=True)
