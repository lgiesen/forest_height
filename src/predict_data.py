import numpy as np
import pandas as pd


def load_img(filepath, features):
    X = np.load(filepath)
    color_channels, width, height = X.shape
    Xr = X.reshape(10,-1).transpose()
    #X_flat = np.reshape(X, (width * height, color_channels))
    if False:
        if "color_channels" in features:
            ceiling = 2000
            X[X > ceiling] = ceiling
            #scale values between 0 and 1
            X = X / ceiling
            color_channels, width, height = X.shape
            X = np.reshape(X, (width*height, color_channels))
        if "NDVI" in features:
            # define columns to prepare NDVI calculation
            X_flat = pd.DataFrame(X_flat, columns = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
            # get NDVI value
            X_NDVI = calculate_ndvi(X_flat)["NDVI"]
            if "color_channels" in features:
                return np.concatenate((X, X_NDVI))
            # reshape data and format it for model predictions
            return np.array(np.reshape(pd.DataFrame(X_NDVI), (-1,1)))
    return X, Xr, (color_channels, width, height)

def pred_img(filepath, features, model, plot_img=True):
    X, Xr, (color_channels, width, height) = load_img(filepath, features)
    pred = model.predict(Xr)
    pred = pred.transpose().reshape(1, width, height)
    if plot_img:
        tree_height_2d = pred[0]
        plot(X)
        plot(tree_height_2d)