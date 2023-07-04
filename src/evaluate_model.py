import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

color = "#01748F"

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mae ** (1/2)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'MAE: {mae}; MSE: {mse}; RMSE: {rmse}; MAPE: {mape}')
    return (mae, mse, rmse, mape)



def feature_importance(model, model_name, cols=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI', 'EVI', 'SAVI', 'IRECI', 's2rep']):
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