from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hour.csv"

TARGET = "cnt"

# Columns that must never be used as features (EDA, section 9)
DROP = ["instant", "casual", "registered", "dteday"]

# Columns dropped as redundant after EDA (EDA, section 10)
EDA_DROP = ["season", "atemp"]

def load_data(drop__yr=True):
    FINAL_DROP = DROP + EDA_DROP + [TARGET]
    if drop__yr:
        FINAL_DROP.append("yr")

    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])

    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("datetime", kind="stable").set_index("datetime")

    df["weathersit"] = df["weathersit"].replace(4, 3)

    df.loc[df["hum"] == 0, "hum"] = np.nan

    y = df[TARGET]
    X = df.drop(columns=FINAL_DROP)
    return X, y
