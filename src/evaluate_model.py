from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mae ** (1/2)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'MAE: {mae}; MSE: {mse}; RMSE: {rmse}; MAPE: {mape}')
    return (mae, mse, rmse, mape)
