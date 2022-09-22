import task_1
import task_2
from preprocess import *
from typing import List

TRAIN_FILE = "waze_data.csv"
TEST_FILE = "waze_take_features.csv"


def load_train(path: str):
    df = load_csv(path, normalize_coords=True, only_tel_aviv=False, remove_nan_subtype=True)
    df = df.sort_values(by=["city"], axis=0, kind='mergesort')
    return df


def main(task1_test_path: str, task2_dates: List[str]):
    np.random.seed(0)
    data = load_train(TRAIN_FILE)

    try:
        task_1.do_task(task1_test_path, TRAIN_FILE, data)
    except:
        print("Task 1 failed")

    try:
        task_2.do_task(task2_dates, data)
    except:
        print("Task 2 failed")


if __name__ == '__main__':
    main("waze_take_features.csv", ["2022-06-05", "2022-06-07", "2022-06-09"])
