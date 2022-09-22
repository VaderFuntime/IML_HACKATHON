import pandas as pd
import numpy as np


def map_datetime_to_timeslot(dt):
    """
    Maps a datetime object to a timeslot
    """
    if (8 <= dt.hour < 10) or (dt.hour == 10 and dt.minute == 0):
        return 1
    elif (12 <= dt.hour < 14) or (dt.hour == 14 and dt.minute == 0):
        return 2
    elif 18 <= dt.hour < 20 or dt.hour == 20 and dt.minute == 0:
        return 3
    else:
        return 0


def load_csv(file_name, normalize_coords, only_tel_aviv=True,
             remove_nan_subtype=False, convert_dates=True, normalize_dates=False):
    """
    Loads a csv file and returns a dataframe
    """
    # keep only the columns we want
    df = pd.read_csv(file_name, usecols=['linqmap_type', 'linqmap_subtype', 'linqmap_city', 'linqmap_street',
                                         'linqmap_roadType', 'update_date', 'x', 'y'])
    if only_tel_aviv:
        df = df[df['linqmap_city'] == 'תל אביב - יפו']

    # rename columns
    df.rename(columns={'linqmap_type': 'type', 'linqmap_subtype': 'subtype', 'linqmap_city': 'city',
                       'linqmap_street': 'street', 'linqmap_roadType': 'roadType'}, inplace=True)
    if convert_dates:
        # convert epoch timestamp to datetime
        df['update_date'] = pd.to_datetime(df['update_date'], unit='ms')
        # add timeslot column
        df['timeslot'] = df['update_date'].apply(map_datetime_to_timeslot)
        df['day_of_week'] = df['update_date'].dt.dayofweek
        df['date'] = df['update_date'].dt.date


    elif normalize_dates:
        df['update_date'] = df['update_date'] - df['update_date'].min()

    if remove_nan_subtype:
        # remove rows subtype missing
        df = df[df['subtype'].notnull()]
    else:
        # Adding missing values: for type jam with missing subtype set subtype to "JAM_HEAVY_TRAFFIC"
        df.loc[df['subtype'].isnull(), 'subtype'] = 'JAM_STAND_STILL'

    if normalize_coords:
        # subtract the minimum from each coordinate
        df['x'] = df['x'] - df['x'].min()
        df['y'] = df['y'] - df['y'].min()

    return df


def create_train_quadruples(df):
    # keep these columns
    df = df[['subtype', 'roadType', 'update_date', 'x', 'y']]
    # create dummy columns for subtype and roadType
    df = pd.get_dummies(df, columns=['subtype', 'roadType'])
    np_arr = df.to_numpy()

    new_arr = np.zeros((np_arr.shape[0] - 4, 4 * np_arr.shape[1]))

    x_labels = np.zeros((np_arr.shape[0] - 4,))
    y_labels = np.zeros((np_arr.shape[0] - 4,))
    for i in range(np_arr.shape[0] - 4):
        new_arr[i] = np.hstack([np_arr[i + j] for j in [0, 1, 2, 3]])
        x_labels[i] = np_arr[i + 4, 1]
        y_labels[i] = np_arr[i + 4, 2]


    return new_arr, x_labels, y_labels, df.iloc[0, :]


def create_test_quadruples(df, first_row):
    # keep these columns
    df = df[['subtype', 'roadType', 'update_date', 'x', 'y']]
    # create dummy columns for subtype and roadType
    df = pd.get_dummies(df, columns=['subtype', 'roadType'])

    # aligning with the train columns
    df, right = df.align(first_row, join="right", axis=1)
    df.fillna(0, inplace=True)
    np_arr = df.to_numpy()

    new_arr = np.zeros((np_arr.shape[0] // 4, 4 * np_arr.shape[1]))
    for i in range(0, np_arr.shape[0], 4):
        new_arr[i // 4] = np.hstack([np_arr[i + j] for j in [0, 1, 2, 3]])

    return new_arr


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 3)
    np.set_options(threshold=np.inf)

    df = load_csv('waze_data.csv', normalize_coords=True, remove_nan_subtype=True, convert_dates=False,
                  normalize_dates=True)

    new_train_arr, x_labels, y_labels, first_row = create_train_quadruples(df)
    # print(df)
    test_df = load_csv("waze_take_features.csv", normalize_coords=False, remove_nan_subtype=False, convert_dates=False,
                       normalize_dates=True)
    # print(test_df)
    test_arr = create_test_quadruples(test_df, first_row)
    # print(test_arr)
