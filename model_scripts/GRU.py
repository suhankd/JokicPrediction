from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsolutePercentageError

import os

Model_GRU = Sequential([

    GRU(units=64,
        input_shape = (1,17),
        return_sequences = True,
        activation = 'tanh'),

    Dropout(0.2),

    GRU(units = 64,
        input_shape = (1,17),
        return_sequences = True,
        activation = 'tanh'),

    Dropout(0.2),

    Dense(1, activation = 'linear')

])

optimizer = Adam(learning_rate = 0.02)
loss = MeanAbsolutePercentageError()

Model_GRU.compile(optimizer = optimizer, loss = loss, metrics=['mae'])