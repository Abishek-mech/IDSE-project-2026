"""Central configuration: paths, random seed, split dates, event calendar."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models"

RAW_URL = ("https://raw.githubusercontent.com/urbanemissionsinfo/AQI_bulletins/"
           "main/data/Processed/AllIndiaBulletins_Master_2024.csv")
RAW_FILE = RAW_DIR / "AllIndiaBulletins_Master_2024.csv"
CITY = "Chennai"
CITY_FILE = RAW_DIR / "chennai_aqi_bulletin.csv"   # small extract kept in the repo

# Chronological split (never shuffled -> no look-ahead leakage)
TRAIN_END = "2021-12-31"      # train : 2015-05-01 .. 2021-12-31
VAL_END = "2022-12-31"        # val   : 2022
                              # test  : 2023-01-01 .. 2024-12-31 (locked until the end)

MAX_INTERP_GAP = 3            # interpolate gaps of at most 3 days
AQI_THRESHOLD = 100           # CPCB: <=100 Satisfactory or better, >100 Moderate or worse

# Approximate Tamil Nadu Deepavali dates (state holiday); flag covers day 0 and day +1 (see features.py).
DEEPAVALI = ["2015-11-10", "2016-10-29", "2017-10-18", "2018-11-06", "2019-10-27",
             "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-10-31"]

CPCB_BINS = [0, 50, 100, 200, 300, 400, 10_000]
CPCB_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
