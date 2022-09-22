from typing import List
import pandas as pd
import distribution_predict
import distribution_preprocess


def do_task(dates: List[str], data: pd.DataFrame):
    feats, labels = distribution_preprocess.load_data(data)
    dist_model = distribution_predict.DistributionPredictor()
    dist_model.fit(feats, labels)

    for date in dates:
        day = pd.to_datetime(date).dayofweek
        preds = dist_model.predict(day)
        pd.DataFrame(preds).to_csv(date+"-predictions.csv",index=False,header=False)