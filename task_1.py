import pandas as pd
from preprocess import load_csv
from subtype_predict import SubtypePredictor
from x_y_regression import regress_coordinates


def load_test(path: str):
    df = load_csv(path, normalize_coords=True, only_tel_aviv=False, remove_nan_subtype=False)
    return df


def do_task(test_path: str, train_path, train_data: pd.DataFrame):
    # Get prediction of next event type
    subtype_pred = SubtypePredictor()
    subtype_pred.fit(train_data)
    test = load_test(test_path)
    type_pred = subtype_pred.predict(test)

    # Get prediction of next x,y
    x_y_df = regress_coordinates(train_path, test_path)
    predictions_df = pd.concat([type_pred, x_y_df], axis=1)

    # save predictions to csv
    predictions_df.rename(columns={"type":"linqmap_type", "subtype":"linqmap_subtype"}, inplace=True)
    predictions_df.to_csv("predictions.csv", index=False)
