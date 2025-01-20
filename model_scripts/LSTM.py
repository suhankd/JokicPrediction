from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsolutePercentageError

import tensorflow as tf

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

Model_LSTM = Sequential([

    LSTM(units = 64, input_shape=(1,17), return_sequences=True),
    Dropout(0.2),

    LSTM(units = 64, input_shape=(1,17), return_sequences=True),
    Dropout(0.2),
    
    Dense(1, activation='linear')
])

optimizer = Adam(learning_rate=0.02)
loss = MeanAbsolutePercentageError()

Model_LSTM.compile(optimizer=optimizer, loss=loss)