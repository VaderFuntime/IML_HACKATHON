from plotly.subplots import make_subplots

from preprocess import *
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

np.set_printoptions(threshold=np.inf)


def compare_regression_models(train_samples, x_labels, y_labels, test_samples, model_type='lasso'):
    if model_type == 'lasso':
        x_model = LassoCV(n_alphas=100, cv=10, max_iter=100000).fit(train_samples, x_labels)
        y_model = LassoCV(n_alphas=100, cv=10, max_iter=100000).fit(train_samples, y_labels)
        print(f"lasso x alpha = {x_model.alpha_}")
        print(f"lasso y alpha = {y_model.alpha_}")

    elif model_type == 'ridge':
        x_model = linear_model.RidgeCV(alphas=np.linspace(1, 1000, 100)).fit(train_samples, x_labels)
        y_model = linear_model.RidgeCV(alphas=np.linspace(1, 1000, 100)).fit(train_samples, y_labels)
        print(f"ridge x alpha = {x_model.alpha_}")
        print(f"ridge y alpha = {y_model.alpha_}")

    else:
        x_model = linear_model.LinearRegression().fit(train_samples, x_labels)
        y_model = linear_model.LinearRegression().fit(train_samples, y_labels)

    return x_model.predict(test_samples), y_model.predict(test_samples)


def get_averages(quad_arr):
    new_arr = np.zeros((quad_arr.shape[0], 2))
    # print(quad_arr[:, [1, 25, 49, 73]])
    new_arr[:, 0] = np.average(quad_arr[:, [1, 25, 49, 73]], axis=1)
    new_arr[:, 1] = np.average(quad_arr[:, [2, 26, 50, 74]], axis=1)
    return new_arr


if __name__ == '__main__':
    # let np print all columns
    np.set_printoptions(threshold=np.inf)

    # x_pred, y_pred = regress_coordinates('waze_data.csv', 'waze_take_features.csv')
    train_data, x_labels, y_labels, first_row = create_train_quadruples(
        load_csv("waze_data.csv", normalize_coords=False, remove_nan_subtype=True,
                 convert_dates=False, normalize_dates=True))

    sample_train, sample_test, label_x_train, label_x_test, label_y_train, label_y_test = \
        model_selection.train_test_split(train_data, x_labels, y_labels, test_size=0.2, random_state=42)

    avg_arr = get_averages(sample_test)
    avg_arr_x_mse = mean_squared_error(avg_arr[:, 0], label_x_test, squared=False)
    avg_arr_y_mse = mean_squared_error(avg_arr[:, 1], label_y_test, squared=False)


    lasso_cv_pred_x, lasso_cv_pred_y = compare_regression_models(sample_train, label_x_train, label_y_train,
                                                                 sample_test,
                                                                 model_type='lasso')
    normal_pred_x, normal_pred_y = compare_regression_models(sample_train, label_x_train, label_y_train, sample_test,
                                                             model_type='linreg')
    ridge_cv_pred_x, ridge_cv_pred_y = compare_regression_models(sample_train, label_x_train, label_y_train,
                                                                 sample_test,
                                                                 model_type='ridge')
    lasso_cv_x_mse = mean_squared_error(label_x_test, lasso_cv_pred_x, squared=False)
    lasso_cv_y_mse = mean_squared_error(label_y_test, lasso_cv_pred_y, squared=False)
    normal_x_mse = mean_squared_error(label_x_test, normal_pred_x, squared=False)
    normal_y_mse = mean_squared_error(label_y_test, normal_pred_y, squared=False)

    ridge_cv_x_mse = mean_squared_error(label_x_test, ridge_cv_pred_x, squared=False)
    ridge_cv_y_mse = mean_squared_error(label_y_test, ridge_cv_pred_y, squared=False)

    print("Lasso CV x MSE: ", lasso_cv_x_mse)
    print("Lasso CV y MSE: ", lasso_cv_y_mse)
    print("Normal x MSE: ", normal_x_mse)
    print("Normal y MSE: ", normal_y_mse)
    print("Ridge CV x MSE: ", ridge_cv_x_mse)
    print("Ridge CV y MSE: ", ridge_cv_y_mse)

    print(f"avg_arr_x_mse = {avg_arr_x_mse}")
    print(f"avg_arr_y_mse = {avg_arr_y_mse}")

    # create a bar chart comparing the MSEs with subplot for each axis
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"X coordinates MSE",
        f"Y coordinates MSE"])
    fig.add_trace(go.Bar(x=['Avg-Coord.', 'LinReg (No Regu.)', 'Lasso', 'Ridge'],
                         y=[avg_arr_x_mse, normal_x_mse, lasso_cv_x_mse, ridge_cv_x_mse], showlegend=False),
                  row=1, col=1)

    # set range to 1300 - 1600
    fig.update_yaxes(range=[1350, 1600], row=1, col=1)
    fig.add_trace(go.Bar(x=['Avg-Coord.', 'LinReg (No Regu.)', 'Lasso', 'Ridge'],
                         y=[avg_arr_y_mse, normal_y_mse, lasso_cv_y_mse, ridge_cv_y_mse], showlegend=False),
                  row=1, col=2)
    # set range to 2000 - 2500
    fig.update_yaxes(range=[2100, 2400], row=1, col=2)
    fig.update_layout(title_text="MSE for X and Y coordinates",
                      xaxis_title="Model",
                      yaxis_title="MSE")

    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="MSE", row=1, col=1)
    fig.update_yaxes(title_text="MSE", row=1, col=2)

    fig.update_layout(title_text="MSE for each model", title_x=0.5)

    fig.show()

