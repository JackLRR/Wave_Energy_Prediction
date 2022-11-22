# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.svm import SVR
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
import xgboost as xgb
from bayes_opt import BayesianOptimization


def BOmodel(n_estimators, max_depth):

    data = pd.read_csv("总数据6.csv", header=0)

    x = data.iloc[0:len(data), 13:21].values
    # 预测浪高
    y = data.iloc[0:len(data), 11].values
    # 预测周期
    y = data.iloc[0:len(data), 12].values
    x = np.array(x)

    future_num = 8
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    def Z_ScoreNormalization(x, mean, sigma):
        x = (x - mean) / sigma
        return x

    for i in range(future_num):
        mean = np.average(x[:, i])
        sigma = np.std(x[:, i])
        for j in range(len(data)):
            x[j, i] = Z_ScoreNormalization(x[j, i], mean, sigma)

    # 数据集比例
    train_volume = int(len(data) * 0.8)
    test_volume = len(data) - train_volume

    train_x = x[0:train_volume, :]
    test_x = x[-test_volume:, :]
    train_y = y[0:train_volume]
    test_y = y[-test_volume:]

    model = ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)  # 随机森林回归,并使用20个决策树
    model.fit(train_x, train_y)

    predicted_data = model.predict(test_x)
    predicted_wave_height = []
    for i in range(len(predicted_data)):
        predicted_wave_height.append(predicted_data[i])


    mse = mean_squared_error(predicted_wave_height, test_y)

    return 1-mse


def RFmodel(n_estimators, max_depth):
    data = pd.read_csv("总数据6.csv", header=0)

    x = data.iloc[0:len(data), 13:21].values
    # 预测浪高
    y = data.iloc[0:len(data), 11].values
    # 预测周期
    y = data.iloc[0:len(data), 12].values
    x = np.array(x)

    future_num = 8
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    def Z_ScoreNormalization(x, mean, sigma):
        x = (x - mean) / sigma
        return x

    for i in range(future_num):
        mean = np.average(x[:, i])
        sigma = np.std(x[:, i])
        for j in range(len(data)):
            x[j, i] = Z_ScoreNormalization(x[j, i], mean, sigma)

    # 数据集比例
    train_volume = int(len(data) * 0.8)
    test_volume = len(data) - train_volume

    train_x = x[0:train_volume, :]
    test_x = x[-test_volume:, :]
    train_y = y[0:train_volume]
    test_y = y[-test_volume:]

    model = ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)  # 随机森林回归,并使用20个决策树
    model.fit(train_x, train_y)

    predicted_data = model.predict(test_x)
    predicted_wave_height = []
    for i in range(len(predicted_data)):
        predicted_wave_height.append(predicted_data[i])

    # 评价模型
    mse = mean_squared_error(predicted_wave_height, test_y)
    rmse = math.sqrt(mean_squared_error(predicted_wave_height, test_y))
    mae = mean_absolute_error(predicted_wave_height, test_y)
    mape = np.mean(np.abs((test_y - predicted_wave_height) / test_y))
    print('mse: %.4f' % mse)
    print('rmse: %.4f' % rmse)
    print('mae: %.4f' % mae)
    print('mape: %.4f' % mape)
    from scipy.stats import pearsonr
    r, p = pearsonr(test_y, predicted_wave_height)
    print("R: %.4f" % r)
    print("R2: %.4f" % r2_score(test_y, predicted_wave_height))

    # 存取预测值
    predicted_wave_height = pd.DataFrame(predicted_wave_height)
    predicted_wave_height.to_csv("RF周期6h预测值.csv", index=0)

rf_bo = BayesianOptimization(
    BOmodel,
    {
        'n_estimators':(10, 200),
        'max_depth':(5, 10)
    }
)
rf_bo.maximize()

rf_bo.maximize()
optimal = rf_bo.max
print(optimal)

params = optimal['params']
n_estimators = params['n_estimators']
max_depth = params['max_depth']

RFmodel(1, 1)