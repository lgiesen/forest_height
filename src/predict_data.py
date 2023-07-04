import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_img(filepath, features):
    X = np.load(filepath)
    color_channels, width, height = X.shape
    Xr = X.reshape(color_channels,-1).transpose()
    #if "color_channels" in features:
    ceiling = 2000
    # counteract high reflection values of clouds
    Xr[Xr > ceiling] = ceiling
    X[X > ceiling] = ceiling
    # counteract significant negative vegetation indices
    Xr[Xr < -1*ceiling] = -1*ceiling
    X[X < -1*ceiling] = -1*ceiling
    assert len(X.shape) == 3
    color_channels, width, height = X.shape
    Xr = np.reshape(Xr, (width*height, color_channels))
    Xr = pd.DataFrame(Xr, columns = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    if "NDVI" in features:
        # get NDVI value
        Xr = calculate_ndvi(Xr)
        if features == ["NDVI"]:
            Xr = Xr["NDVI"]
    if "VI" in features:
        Xr = calculate_VIs(Xr)
        if features == ["VI"]:
            vi_cols = ["EVI", "SAVI", "IRECI", "s2rep"]
            Xr = X_VI[vi_cols]
        elif features == ["NDVI", "VI"]:
            Xr = Xr[["NDVI", "EVI", "SAVI", "IRECI", "s2rep"]]
    scaler = StandardScaler()
    X, Xr = scaler.fit_transform(Xr), scaler.fit_transform(X)
    return X, Xr, (color_channels, width, height)

def pred_img(filepath, features, model, plot_img=True, scale=True):
    pred = model.predict(Xr)
    pred = pred.transpose().reshape(1, width, height)
    if plot_img:
        tree_height_2d = pred[0]
        plot(X)
        plot(tree_height_2d)