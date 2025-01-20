import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Dense
from tensorflow.keras.losses import MeanAbsolutePercentageError
from sklearn.preprocessing import StandardScaler

Model_CNN = Sequential([

    Conv1D(filters=64, 
           kernel_size=2, 
           activation='relu', 
           input_shape = (17,1)),

    MaxPooling1D(pool_size=2),

    Flatten(),
    
    Dense(50, activation='relu'),

    Dense(1, activation='linear')
])


optimizer = Adam(learning_rate=0.02)
loss = MeanAbsolutePercentageError()

Model_CNN.compile(optimizer = optimizer,loss = loss, metrics = ['mae'])