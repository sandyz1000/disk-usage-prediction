# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def mape(y_pred, y_true):
    """MAPE (Mean Absolute Percentage Error) calculation function"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def create_plot(y_pred, y_true):
    """
    Creates a graph of the actual values, linear regression and model predictions
    using the last value
    :param y_pred:
    :param y_true:
    :return:
    """
    plt.ylabel(u'Rate')
    plt.xlabel(u'Periods)')
    reg_val, = plt.plot(y_pred, color='b', label=u'Linear Regression')
    true_val, = plt.plot(y_true, color='g', label=u'True Values')
    plt.xlim([0, len(y_true)])
    plt.legend(handles=[true_val, reg_val])
    # plt.show()
    plt.savefig("time-series.png")


def create_subplot():
    fig, axes = plt.subplot()


def create_features(data):
    """Creates a feature matrix with values from previous 6 days"""
    x_data, y_data = [], []
    for d in range(6, data.shape[0]):
        x = data.iloc[d - 6:d, 2].values.ravel()
        y = data.iloc[d, 2]
        x_data.append(x)
        y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data


def create_model(x_data, y_data):
    """Lists to store the predictions of the models"""
    y_pred = []
    y_pred_last = []
    y_pred_ma = []
    y_true = []

    # Iterate over the time series creating a new model each month
    end = y_data.shape[0]
    for i in range(200, end):
        x_train = x_data[:i, :]
        y_train = y_data[:i]

        x_test = x_data[i, :]
        y_test = y_data[i]

        model = LinearRegression(normalize=True)
        model.fit(x_train, y_train)

        y_pred.append(model.predict(x_test.reshape(1, -1))[0])
        y_pred_last.append(x_test[-1])
        y_pred_ma.append(x_test.mean())
        y_true.append(y_test)

    # Transforms the lists into numpy arrays
    y_pred = np.array(y_pred)
    y_pred_last = np.array(y_pred_last)
    y_pred_ma = np.array(y_pred_ma)
    y_true = np.array(y_true)
    return y_pred, y_pred_last, y_pred_ma, y_true


if __name__ == '__main__':
    # Loading the data
    DATA_SET = os.path.join(os.path.abspath(""), "data", "diskspace_data.csv")

    df = pd.read_csv(DATA_SET, sep=",")
    df.columns = ['SERVER', 'DRIVE', 'TOTAL', 'USED', 'PERCENTAGE_USAGE', 'DATETIME']

    SERVER_SVRINCCGSDB1 = df['SERVER'].apply(lambda x: x == 'SVRINCCGSDB1')
    DRIVE_C = df['DRIVE'].apply(lambda x: x == "C:")
    workset = df[SERVER_SVRINCCGSDB1 & DRIVE_C]

    workset.drop(['SERVER', 'DRIVE'], axis=1, inplace=True)
    parsedate = workset['DATETIME']\
        .apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').date() !=
                         datetime(2017, 3, 17).date())

    workset = workset[parsedate]

    x_data, y_data = create_features(workset)
    y_pred, y_pred_last, y_pred_ma, y_true = create_model(x_data, y_data)

    # Print errors
    print('\nMean Absolute Percentage Error')
    print('MAPE Linear Regression', mape(y_pred, y_true))
    print('MAPE Last Value Benchmark', mape(y_pred_last, y_true))
    print('MAPE Moving Average Benchmark', mape(y_pred_ma, y_true))

    print('\nMean Absolute Error')
    print('MAE Linear Regression', mean_absolute_error(y_pred, y_true))
    print('MAE Last Value Benchmark', mean_absolute_error(y_pred_last, y_true))
    print('MAE Moving Average Benchmark', mean_absolute_error(y_pred_ma, y_true))

    create_plot(y_pred, y_true)
