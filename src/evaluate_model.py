import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

color = "#01748F"

def evaluate_model(y_test, y_pred):
    """
    Evaluates a model based on its test set prediction

    Parameters
    ----------
    y_test: pandas.DataFrame
    y_pred: pandas.DataFrame

    Returns
    -------
    Errors (Tuple of Float)
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mae ** (1/2)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'MAE: {mae}; MSE: {mse}; RMSE: {rmse}; MAPE: {mape}')
    return (mae, mse, rmse, mape)

def train_evaluate_model(model, dataset=["color_channels", "color_channels_ndvi", "ndvi", "all"]):
    """
    Train and evaluate a model on specified datasets

    Parameters
    ----------
    model: sklearn.ensemble.*
    dataset: Array of Strings

    Returns
    -------
    None, just prints out errors of each dataset
    """
    for ds in dataset:
        print(ds)
        X_train, y_train, X_test, y_test = load_data(ds)
        # train model
        model.fit(X_train, y_train)
        # predict test set
        y_pred = model.predict(X_test)
        # evaluate model
        mae, mse, rmse, mape = evaluate_model(y_test, y_pred)


def feature_importance(model, model_name, cols=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI', 'EVI', 'SAVI', 'IRECI', 's2rep']):
    """
    Visualize feature importance of regression model

    Parameters
    ----------
    model: sklearn.ensemble.*
    model_name: String
    cols: Array of Strings

    Returns
    -------
    None, just prints out feature importances and plots them in a bar graph
    """
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar(cols, importance, color=color)
    plt.xlabel("Features")
    plt.ylabel("Feature Importance")
    plt.title(f"Feature Importance of {model_name} Regression")
    plt.show()

def pred_vs_true(model, model_name, ds="all"):
    """
    Visualize predictions and compare them to the labeled data

    Parameters
    ----------
    model: sklearn.ensemble.*
    model_name: String
    ds: Array of Strings

    Returns
    -------
    None, just prints out errors of each dataset
    """
    # get necessary data
    X_train, y_train, X_test, y_test = load_data(ds)
    y_pred = model.predict(X_test)
    
    # visualize predictions vs. true labels
    fig = plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_test, color=color, alpha=0.5)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.plot([-1,75], [-1, 75], 'k--')
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.xlim([-1,75])
    plt.ylim([-1,75])
    plt.title(f"{model_name} Regression: Prediction vs. Labels")
    plt.show()

    # only the NDVI channel is plotted on the x-axis
    # because 11-dimensional data cannot be visualized for humans
    channel = 10 if X_test.shape[1] > 9 else 0
    fig, ax = plt.subplots()
    plt.scatter(X_test[:,10], y_test, 10, color='black')
    plt.scatter(X_test[:,10], y_pred, 10, color=color)
    plt.title(f'{model_name} Regression: NDVI and Forest Height')
    if channel == 10:
        plt.xlabel('NDVI Value')
    plt.ylabel('Forest Height')
    ax.legend(("True Value", "Prediction"), loc='upper left')
    plt.show()

def save_model(model, modelname, ds):
    train_evaluate_model(model, [ds])
    joblib.dump(model, f'forest_height/models/{modelname}.joblib')
    # load model with:
    # model = joblib.load("forest_height/models/{model_name}.joblib")