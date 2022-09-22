from sklearn import linear_model
from preprocess import *
import pandas as pd

RIDGE_ALPHA = 197
LASSO_X_ALPHA = 34459461
LASSO_Y_ALPHA = 1077981815


def regress_coordinates(train_file, test_file):
    train_data, x_labels, y_labels, first_row = create_train_quadruples(
        load_csv(train_file, normalize_coords=False, remove_nan_subtype=True,
                 convert_dates=False, normalize_dates=True))
    test_data = create_test_quadruples(
        load_csv(test_file, normalize_coords=False, remove_nan_subtype=False, convert_dates=False,
                 normalize_dates=True), first_row)

    x_model = linear_model.Ridge(RIDGE_ALPHA).fit(train_data, x_labels)
    y_model = linear_model.Ridge(RIDGE_ALPHA).fit(train_data, y_labels)

    x_y_df = pd.DataFrame(columns=['x', 'y'])
    x_y_df['x'] = x_model.predict(test_data)
    x_y_df['y'] = y_model.predict(test_data)
    return x_y_df
