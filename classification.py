import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import LabelEncoder

def sale_price_replace(house_prices):
    df = house_prices.copy()
    cluster_calc = df[['GrLivArea', 'SalePrice','1stFlrSF','GarageArea']]
    cluster_calc.dropna()
    cluster_set = breif_clustering(cluster_calc, 3)
    
    tem = df[df.columns]
    tem['SpThird'] = cluster_set['Cluster']
    tem.pop('SalePrice')
    return tem


def breif_clustering(X, n_clusters):

    X_pca = PCA(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X_pca)

    X['Cluster'] = km.fit_predict(X_pca)
    centroides = km.cluster_centers_
    return X

def metrics_and_cm(y_pred, y_test):
    # Presicion
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
def trans_categorical(df):
    df = df.copy()  # Evitar modificar el dataframe original
    
    # Definir las variables categóricas a transformar
    ordinal_mappings = {
        'ExterQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'FireplaceQu': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3},
        'PoolQC': {'Fa': 1, 'Gd': 2, 'Ex': 3},
        'Fence': {'MnWw': 1, 'MnPrv': 2, 'GdWo': 3, 'GdPrv': 4}
    }
    
    # Para las variables ordinales, utilizamos el mapeo definido
    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping).fillna(0)
    
    # Variables nominales -> Label Encoding
    nominal_cols = [
        'MSZoning', 'Street', 'LandContour', 'Utilities', 'LandSlope',
        'Condition1', 'Condition2', 'RoofMatl', 'BsmtCond', 'BsmtExposure',
        'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageQual',
        'GarageCond', 'PavedDrive', 'MiscFeature', 'SaleType', 'SaleCondition'
    ]
    
    for col in nominal_cols:
        if df[col].dtype == 'object':  # Solo aplicar si es tipo 'object'
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))  # Convertir a int
    
    return df