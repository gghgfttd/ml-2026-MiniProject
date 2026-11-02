from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hour.csv"

TARGET = "cnt"

DROP = ["instant", "casual", "registered", "dteday"]

#"atemp" (r = 0.99 з temp), "season" (груба версія mnth)
EDA_DROP = []

def load_data():

    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])

    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("datetime", kind="stable").set_index("datetime")

    df.loc[df["hum"] == 0, "hum"] = np.nan

    y = df[TARGET]
    X = df.drop(columns=DROP + EDA_DROP + [TARGET])
    return X, y

