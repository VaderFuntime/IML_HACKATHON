from typing import *

import pandas as pd
from pandas import DataFrame

def load_data(df: DataFrame):
    feat = df
    feat["counts"] = feat["type"]
    feat = feat.groupby(["date", "timeslot","type"],as_index=False)["counts"].count()
    feat["dayofweek"] = pd.to_datetime(feat["date"]).dt.dayofweek
    feat = pd.concat([feat, pd.get_dummies(feat["type"])], axis=1)
    labels = feat["counts"]
    feat = feat.drop(["date","counts","type"],axis=1)

    return feat.to_numpy(), labels.to_numpy()




