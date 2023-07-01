import numpy as np
import pandas as pd


def load_data(data_type):
    forest_data = "forest_height/data/"
    X_train = np.array(pd.read_pickle(f"{forest_data}{data_type}/X_train.pkl"))
    y_train = np.array(pd.read_pickle(f"{forest_data}{data_type}/y_train.pkl"))
    X_test = np.array(pd.read_pickle(f"{forest_data}{data_type}/X_test.pkl"))
    y_test = np.array(pd.read_pickle(f"{forest_data}{data_type}/y_test.pkl"))
    return (X_train, y_train, X_test, y_test)