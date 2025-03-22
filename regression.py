import scipy.stats as stats
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

def drop_many_nulls(df):
    df = df.copy()  # Evitar modificar el dataframe original
    
    # Eliminar las variables que no queremos en el análisis de clusters
    drop_columns = [
        'Id', 'PoolArea', 'MiscVal', 'BsmtFinSF2', 'BsmtFinSF1', 'MasVnrArea',
        'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Alley', 'ExterCond',
        'BsmtHalfBath', 'KitchenAbvGr', 'PoolQC', 'Fence', 'MiscFeature', 'MiscFeature',
        'FireplaceQu', 'MasVnrType', 
    ]
    df = df.drop(columns=drop_columns, errors='ignore')
    
    return df

def performance_metrics(y,y_pred):
    rmse = root_mean_squared_error(y, y_pred)
    mae_knn = mean_absolute_error(y, y_pred)
    mse_knn = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"RMSE en test: {rmse}")
    print(f"MAE en test: {mae_knn}")
    print(f"MSE en test: {mse_knn}")
    print(f"R² en test: {r2}")
