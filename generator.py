import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from processing import processing

dfs = processing(f'Data')

scaler = StandardScaler()

all_data = np.concatenate([dfs[f'Data\\{year}'].drop('PtsNextGame', axis=1).to_numpy() for year in range(2021, 2025)], axis=0)
scaler.fit(all_data)

# LSTM

from model_scripts.LSTM import Model_LSTM

for year in range(2021, 2024):
    
    X = (dfs[f'Data\\{year}'].drop('PtsNextGame', axis=1)).to_numpy()
    y =(dfs[f'Data\\{year}']['PtsNextGame']).to_numpy()
    
    X = X.reshape((X.shape[0], X.shape[1]))
    y = y.reshape(y.shape[0], 1)

    X_normalized = scaler.transform(X)

    num_features = X_normalized.shape[1]

    X_normalized = X_normalized.reshape((X_normalized.shape[0], 1, num_features))  # Shape: (samples, time_steps, features)
    
    Model_LSTM.fit(X_normalized, y, epochs=100)

Model_LSTM.save('Models/Model_LSTM.keras')    


# GRU

from model_scripts.GRU import Model_GRU

for year in range(2021, 2024):

    X = (dfs[f'Data\\{year}'].drop('PtsNextGame', axis=1)).to_numpy()
    y =(dfs[f'Data\\{year}']['PtsNextGame']).to_numpy()

    X = scaler.transform(X)

    X = X.reshape((X.shape[0], 1, X.shape[1]))
    y = y.reshape(y.shape[0], 1)

    Model_GRU.fit(X, y, epochs=100)

Model_GRU.save('Models/Model_GRU.keras')

# CNN

from model_scripts.CNN import Model_CNN

for year in range(2021, 2024):
    X = dfs[f'Data\\{year}'].drop('PtsNextGame', axis=1).to_numpy()
    y = dfs[f'Data\\{year}']['PtsNextGame'].to_numpy()

    X = scaler.transform(X)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape(y.shape[0], 1)

    Model_CNN.fit(X, y, epochs=100)

Model_CNN.save('Models/Model_CNN.keras')

# XGBoost

from xgboost import XGBRegressor
Model_XGB = XGBRegressor()

for year in range(2021, 2024):
    X = dfs[f'Data\\{year}'].drop('PtsNextGame', axis=1).to_numpy()
    y = dfs[f'Data\\{year}']['PtsNextGame'].to_numpy()

    X = scaler.transform(X)

    X = X.reshape((X.shape[0], X.shape[1]))
    y = y.reshape(y.shape[0], 1)

    Model_XGB.fit(X, y)

Model_XGB.save_model('Models/Model_XGB.bin')