import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_img(filepath, features=["NDVI", "VI"], scale=True):
    """
    Loads image and generates vegetation features if specified

    Parameters
    ----------
    filepath: String
    features: Array of Strings

    Returns
    -------
    Original image (pandas.DataFrame), adjusted image (pandas.DataFrame), shape (Integer, Integer, Integer)
    """
    X = np.load(filepath)
    color_channels, width, height = X.shape
    Xr = X.reshape(color_channels,-1).transpose()
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
            Xr = Xr[vi_cols]
        elif features == ["NDVI", "VI"]:
            Xr = Xr[["NDVI", "EVI", "SAVI", "IRECI", "s2rep"]]
    ceiling = 2000
    # counteract high reflection values of clouds
    Xr[Xr > ceiling] = ceiling
    X[X > ceiling] = ceiling
    # counteract significant negative vegetation indices
    Xr[Xr < -1*ceiling] = -1*ceiling
    X[X < -1*ceiling] = -1*ceiling
    if scale:
        scaler = StandardScaler()
        Xr = scaler.fit_transform(Xr)
    return X, Xr, (color_channels, width, height)

def pred_img(filepath, features, model, plot_img=True, scale=True):
    """
    Predicts an image with specified features and accordingly trained model

    Parameters
    ----------
    filepath: String
    features: Array of Strings
    model: sklearn.ensemble.*
    plot_img: Boolean
    scale: Boolean

    Returns
    -------
    satellite image (pandas.Dataframe), prediction (pandas.Dataframe)
    """
    X, Xr, (color_channels, width, height) = load_img(
        filepath=filepath, 
        features=features,
        scale=scale)
    pred = model.predict(Xr)
    pred = pred.transpose().reshape(1, width, height)
    if plot_img:
        tree_height_2d = pred[0]
        plot(X)
        plot(tree_height_2d)
    return X, pred