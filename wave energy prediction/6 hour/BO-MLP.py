# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from tcn import TCN
from tensorflow.keras.layers import Dropout, Dense, LSTM, GRU, Bidirectional
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import math
import time
from bayes_opt import BayesianOptimization
from tensorflow.python.keras.layers import Attention


def BOmodel(units):

    data = pd.read_csv("总数据6.csv", header=0)

    x = data.iloc[0:len(data), 13:21].values

    # 预测浪高
    y = data.iloc[0:len(data), 11].values
    # 预测周期
    y = data.iloc[0:len(data), 12].values

    x = np.array(x)

    future_num = 8
    units = int(units)

    def Z_ScoreNormalization(x, mean, sigma):
        x = (x - mean) / sigma
        return x

    for i in range(future_num):
        mean = np.average(x[:, i])
        sigma = np.std(x[:, i])
        for j in range(len(data)):
            x[j, i] = Z_ScoreNormalization(x[j, i], mean, sigma)

    # 数据集比例
    train_volume = int(len(data) * 0.64)
    val_volume = int(len(data) * 0.16)
    test_volume = len(data) - train_volume - val_volume

    train_x = x[0:train_volume, :]
    val_x = x[train_volume:train_volume+val_volume, :]
    test_x = x[-test_volume:, :]
    train_y = y[0:train_volume]
    val_y = y[train_volume:train_volume+val_volume]
    test_y = y[-test_volume:]

    x_train, y_train = np.array(train_x), np.array(train_y)

    x_val, y_val = np.array(val_x), np.array(val_y)

    x_test, y_test = np.array(test_x), np.array(test_y)

    model = tf.keras.Sequential([
        Dense(units, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])

    cp_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=48, epochs=100, validation_data=(x_val, y_val), validation_freq=1,
                        callbacks=cp_callback, verbose=0)


    predicted_data = model.predict(x_test)


    predicted_wave = []
    for i in predicted_data:
        for j in i:
            predicted_wave.append(j)

    predicted_wave_height = []
    for i in range(len(predicted_wave)):
        predicted_wave_height.append(predicted_wave[i])

    # 评价模型
    mse = mean_squared_error(predicted_wave_height, y_test)
    return 1-mse

def LSTMmodel(units):

    data = pd.read_csv("总数据6.csv", header=0)

    x = data.iloc[0:len(data), 13:21].values

    # 预测浪高
    y = data.iloc[0:len(data), 11].values
    # 预测周期
    y = data.iloc[0:len(data), 12].values

    x = np.array(x)

    future_num = 8
    units = int(units)

    def Z_ScoreNormalization(x, mean, sigma):
        x = (x - mean) / sigma
        return x

    for i in range(future_num):
        mean = np.average(x[:, i])
        sigma = np.std(x[:, i])
        for j in range(len(data)):
            x[j, i] = Z_ScoreNormalization(x[j, i], mean, sigma)

    # 数据集比例
    train_volume = int(len(data) * 0.64)
    val_volume = int(len(data) * 0.16)
    test_volume = len(data) - train_volume - val_volume

    train_x = x[0:train_volume, :]
    val_x = x[train_volume:train_volume+val_volume, :]
    test_x = x[-test_volume:, :]
    train_y = y[0:train_volume]
    val_y = y[train_volume:train_volume+val_volume]
    test_y = y[-test_volume:]

    x_train, y_train = np.array(train_x), np.array(train_y)

    x_val, y_val = np.array(val_x), np.array(val_y)

    x_test, y_test = np.array(test_x), np.array(test_y)

    model = tf.keras.Sequential([
        Dense(units, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])

    cp_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=48, epochs=100, validation_data=(x_val, y_val), validation_freq=1,
                        callbacks=cp_callback)

    predicted_data = model.predict(x_test)


    predicted_wave = []
    for i in predicted_data:
        for j in i:
            predicted_wave.append(j)

    predicted_wave_height = []
    for i in range(len(predicted_wave)):
        predicted_wave_height.append(predicted_wave[i])

    # 评价模型
    mse = mean_squared_error(predicted_wave_height, y_test)
    rmse = math.sqrt(mean_squared_error(predicted_wave_height, y_test))
    mae = mean_absolute_error(predicted_wave_height, y_test)
    mape = np.mean(np.abs((y_test - predicted_wave_height) / y_test))
    print('mse: %.4f' % mse)
    print('rmse: %.4f' % rmse)
    print('mae: %.4f' % mae)
    print('mape: %.4f' % mape)
    from scipy.stats import pearsonr
    r, p = pearsonr(y_test, predicted_wave_height)
    print("R: %.4f" % r)
    print("R2: %.4f" % r2_score(y_test, predicted_wave_height))

    # 存取预测值
    predicted_wave_height = pd.DataFrame(predicted_wave_height)
    predicted_wave_height.to_csv("MLP周期6h预测值.csv", index=0)

lstm_bo = BayesianOptimization(
    BOmodel,
    {
        'units':(2, 128),
    }
)
lstm_bo.maximize()
optimal = lstm_bo.max
params = optimal['params']
units = params['units']

LSTMmodel(1)