#!/usr/bin/python3.7
import time
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import numpy as np
from numpy import vstack
import talib as ta
from numpy import array
from numpy import hstack
from keras.models import  Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
import h5py
import pandas as pd
from keras.losses import mean_squared_error
import numpy as np
import math
import pytz
from datetime import datetime
import datetime as dt
import sqlalchemy
from binance import BinanceSocketManager
import mplfinance as mpf
import matplotlib.pyplot as plt
from binance.enums import  *
import subprocess
import os
import sys
from talib import MACD
from talib import MACDEXT
from talib import RSI
from talib import SMA
from pandas import DataFrame
from talib import SMA
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from talib import CDLDARKCLOUDCOVER
from talib import MFI
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gc
import keras as k

#Initiate client API
api_key = "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
api_secret = "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ "
client = Client(api_key,api_secret)
print(client.ping(), flush=True)


# Define a nested function 'forloop' that generates a model name based on a number and a model number
def forloop(number, modelnumber, modelname):
    # Get the current date and time
    pr = datetime.today()

    # Add 30 to the input number
    num2 = number + 30

    # Calculate the start and end dates based on the input number
    start_date = pr - timedelta(days=num2)
    end_date = pr - timedelta(days=number)

    # Convert the start and end dates to strings
    d = str(start_date)
    r = d[:10]
    d2 = str(end_date)
    r2 = d2[0:10]

    # Create a name for the model based on the model number and the start and end dates
    name = modelname + 'Model_' + str(modelnumber) + '-' + r + '-' + r2 + '_Week_' + str(modelnumber) + '.h5'

    # Return the model name
    return name
#Extract four names using a model name
def extract_four_names(modelname):
        # Generate four model names using different numbers and model numbers
        namemodel1 = forloop(21, 1, modelname)
        namemodel2 = forloop(14, 2, modelname)
        namemodel3 = forloop(7, 3, modelname)
        namemodel4 = forloop(0, 4, modelname)

        # Return all four model names
        return namemodel1, namemodel2, namemodel3, namemodel4

# Split Sequence used in splitting input into a machine learninig model
def split_sequences(sequences, n_steps):
            X, y = list(), list()
            for i in range(len(sequences)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the dataset
                if end_ix > len(sequences):
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)

#Prepare data for input into a machine learning model
def prepare(c_four_hour, c_four_hourBTC):

            # Create a new column based on whether the closing price is higher or lower than the opening price
            c_four_hour[:, 11:12] = np.array(
                [(0 if x >= y else 1) for x, y in zip(c_four_hour[:, 1:2], c_four_hour[:, 4:5])]).reshape(-1, 1)
            fifty = 50
            thirty = 30
            onetwenty = 120
            # Calculate various technical indicators
            oopen, loow, highh, voluu, closee = [np.array(c_four_hour[:, i:i + 1].ravel(), dtype=float) for i in
                                                 range(4, 9)]
            ooopen = np.array(c_four_hourBTC[:, 4:5].ravel(), dtype=float)
            real = SMA(oopen, timeperiod=50)[120:]
            realb = SMA(ooopen, timeperiod=50)[120:]
            real30 = SMA(oopen, timeperiod=30)[120:]
            macd, signal, hist = ta.MACD(oopen, fastperiod=20, slowperiod=87, signalperiod=37)
            r14 = ta.RSI(oopen, timeperiod=14)
            rb14 = ta.RSI(ooopen, timeperiod=14)
            r2 = ta.RSI(oopen, timeperiod=2)
            rb2 = ta.RSI(ooopen, timeperiod=2)
            r6 = ta.RSI(oopen, timeperiod=6)
            rb6 = ta.RSI(ooopen, timeperiod=6)
            r8 = ta.RSI(oopen, timeperiod=8)
            rb8 = ta.RSI(ooopen, timeperiod=8)
            r11 = ta.RSI(oopen, timeperiod=11)

            # More technical indicators
            macd1, signal1, hist1 = ta.MACD(oopen, fastperiod=12, slowperiod=26, signalperiod=9)
            integer = MFI(highh, loow, closee, voluu, timeperiod=14)
            ppo1 = ta.PPO(oopen, fastperiod=12, slowperiod=26, matype=0)
            max1 = ta.MAX(oopen, timeperiod=30)
            ema1 = ta.EMA(oopen, timeperiod=50)[120:]
            ema2 = ta.EMA(oopen, timeperiod=10)[120:]
            ema3 = ta.EMA(oopen, timeperiod=80)[120:]
            ema4 = ta.EMA(oopen, timeperiod=26)[120:]
            emafive = ta.EMA(oopen, timeperiod=5)[120:]
            ema5 = ta.EMA(oopen, timeperiod=8)[120:]

            # Stack all these features horizontally to create a single numpy array
            new_c_four_hour = hstack((macd[120:, None], signal[120:, None], hist[120:, None], real[:, None],
                                            real30[:, None], integer[120:, None], r14[120:, None], r2[120:, None],
                                            r6[120:, None], macd1[120:, None], signal1[120:, None], hist1[120:, None],
                                            ema1[:, None], ema2[:, None],
                                            ema3[:, None], ema4[:, None], ema5[:, None], max1[120:, None],
                                            ppo1[120:, None], r8[120:, None], r11[120:, None], rb2[120:, None],
                                            rb14[120:, None], rb6[120:, None], rb8[120:, None], realb[:, None],
                                            r14[120:, None], emafive[:, None],
                                            integer[120:, None],r14[120:, None],
                                            c_four_hourBTC[onetwenty:, 5:6],
                                            c_four_hour[onetwenty:, 1:6],
                                            c_four_hour[onetwenty:, 11:12]))


            new_c_four_hour_mayeb = hstack(( new_c_four_hour[:-1,:],new_c_four_hour[1:,36:37]))
            pr = new_c_four_hour_mayeb.astype(np.float)
            prnew = pr[:,:]
            # Standardize the data
            scs = [StandardScaler() for _ in range(39)]
            prnew = new_c_four_hour_mayeb.astype(np.float)

            for i in range(39):
                prnew[:, i:i + 1] = scs[i].fit_transform(prnew[:, i:i + 1])

            # Prepare the input features and target variable for your model
            n_steps = 170
            wee = hstack((prnew[:, 6:7], prnew[:, 37:38]))
            X, y = split_sequences(prnew, n_steps)
            X = X.astype(np.float)
            y = y.astype(np.float)
            n_features = X.shape[2]
            return X, y, n_steps, n_features, *scs, prnew

#Prepare testing data for a machine learning model
def prepare_test(c_four_hour, c_four_hour_TESTBTC, *scs):


            # Create a new column based on whether the closing price is higher or lower than the opening price
            c_four_hour[:, 11:12] = np.array([(0 if x >= y else 1) for x, y in zip(c_four_hour[:, 1:2], c_four_hour[:, 4:5])]).reshape(-1, 1)

            # Extract necessary data from the input arrays
            oopen, loow, highh, voluu, closee = [np.array(c_four_hour[:, i:i + 1].ravel(), dtype=float) for i in range(4, 9)]
            ooopen = np.array(c_four_hour_TESTBTC[:, 4:5].ravel(), dtype=float)
            fifty = 50
            thirty = 30
            onetwenty = 120
            # Calculate various technical indicators
            real = SMA(oopen, timeperiod=50)[120:]
            realb = SMA(ooopen, timeperiod=50)[120:]
            real30 = SMA(oopen, timeperiod=30)[120:]
            macd, signal, hist = ta.MACD(oopen, fastperiod=20, slowperiod=87, signalperiod=37)
            r14 = ta.RSI(oopen, timeperiod=14)
            rb14 = ta.RSI(ooopen, timeperiod=14)
            r2 = ta.RSI(oopen, timeperiod=2)
            rb2 = ta.RSI(ooopen, timeperiod=2)
            r6 = ta.RSI(oopen, timeperiod=6)
            rb6 = ta.RSI(ooopen, timeperiod=6)
            r8 = ta.RSI(oopen, timeperiod=8)
            rb8 = ta.RSI(ooopen, timeperiod=8)
            r11 = ta.RSI(oopen, timeperiod=11)

            # More technical indicators
            macd1, signal1, hist1 = ta.MACD(oopen, fastperiod=12, slowperiod=26, signalperiod=9)
            integer = MFI(highh, loow, closee, voluu, timeperiod=14)
            ppo1 = ta.PPO(oopen, fastperiod=12, slowperiod=26, matype=0)
            max1 = ta.MAX(oopen, timeperiod=30)
            ema1 = ta.EMA(oopen, timeperiod=50)[120:]
            ema2 = ta.EMA(oopen, timeperiod=10)[120:]
            ema3 = ta.EMA(oopen, timeperiod=80)[120:]
            ema4 = ta.EMA(oopen, timeperiod=26)[120:]
            emafive = ta.EMA(oopen, timeperiod=5)[120:]
            ema5 = ta.EMA(oopen, timeperiod=8)[120:]

            # Stack all these features horizontally to create a single numpy array

            new_c_four_hour = hstack((macd[120:, None], signal[120:, None], hist[120:, None],real[:, None], real30[:, None], integer[120:, None], r14[120:, None], r2[120:, None],
                                            r6[120:, None], macd1[120:, None], signal1[120:, None], hist1[120:, None],
                                            ema1[:, None], ema2[:, None],
                                            ema3[:, None], ema4[:, None], ema5[:, None], max1[120:, None],
                                            ppo1[120:, None], r8[120:, None], r11[120:, None], rb2[120:, None],
                                            rb14[120:, None], rb6[120:, None], rb8[120:, None], realb[:, None],
                                            r14[120:, None], emafive[:, None],
                                            integer[120:, None],r14[120:, None],
                                            c_four_hour_TESTBTC[onetwenty:, 5:6],
                                            c_four_hour[onetwenty:, 1:6],
                                            c_four_hour[onetwenty:, 11:12]))

            new_c_four_hour_mayeb = hstack((new_c_four_hour[:-1, :], new_c_four_hour[1:, 36:37]))
            # Standardize the data using the provided scalers
            prnew = new_c_four_hour_mayeb.astype(np.float)

            for i in range(39):
                prnew[:, i:i + 1] = scs[i].transform(prnew[:, i:i + 1])

            # Prepare the input features and target variable for your model
            n_steps = 170
            wee = hstack((prnew[:, 6:7], prnew[:, 37:38]))

            X_test, y_test = split_sequences(prnew, n_steps)

            X_test, y_test = X_test.astype(np.float), y_test.astype(np.float)

            return X_test, y_test, new_c_four_hour, new_c_four_hour_mayeb, prnew

#Extract list of scales
def importinung(c_four_hour,c_four_hourBTC):
    # Call the 'prepare' function with the input parameters 'c_four_hour' and 'c_four_hourBTC'
    # The 'prepare' function is expected to return multiple values including training data, features, steps, and scales
    X_train, y_train, n_steps, n_features, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13,\
    sc14, sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24, sc25, sc26, sc27, sc28, sc29, sc30, sc31, \
    sc32, sc33, sc34, sc35, sc36, sc37, sc38, sc39, crnew = prepare(
        c_four_hour, c_four_hourBTC)

    # Create a list of all the scales returned by the 'prepare' function
    list_of_scales = [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13,
    sc14, sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24, sc25, sc26,
    sc27, sc28, sc29, sc30, sc31,sc32, sc33, sc34, sc35, sc36, sc37, sc38, sc39]

    # Return the list of scales
    return list_of_scales

# This is a decorator function that modifies the behavior of the Make_Prediction function
def Wrap_Make_prediction(Make_Prediction):
    # The inner function is a wrapper around the Make_Prediction function
    def inner(COINNAME, COINSCALEDLIST, code=1):
        # Call the Make_Prediction function and capture its three return values
        t1, t2, t3 = Make_Prediction(COINNAME, COINSCALEDLIST, code)
        # Only return the first value
        return t1
    # Return the modified function
    return inner


def Make_Prediction(COINNAME, COINSCALEDLIST, Date = 0, code = 0):
            # Convert the scaled list to a numpy array
            R = np.array(COINSCALEDLIST)
            # Unpack the array into individual variables
            sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, \
            sc14, sc15, sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24, sc25, sc26, \
            sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34, sc35, sc36, sc37, sc38, \
            sc39 = R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9], R[10], R[11], R[12], R[13], \
                   R[14], R[15], R[16], R[17], R[18], R[19], R[20], R[21], R[22], R[23], R[24], R[25], R[26], \
                   R[27], R[28], R[29], R[30], R[31], R[32], R[33], R[34], R[35], R[36], R[37], R[38]
            # Depending on the code, fetch historical data
            if (code == 0):
                c_four_hour_TEST = client.get_historical_klines(COINNAME, Client.KLINE_INTERVAL_30MINUTE, Date[0], Date[1],
                                                                klines_type=HistoricalKlinesType.FUTURES)
                c_four_hour_TESTBTC = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_30MINUTE, Date[0],
                                                               Date[1],
                                                               klines_type=HistoricalKlinesType.FUTURES)
            else:
                c_four_hour_TEST = client.get_historical_klines(COINNAME, Client.KLINE_INTERVAL_30MINUTE, '146 HOUR ago UTC ',
                                                        klines_type=HistoricalKlinesType.FUTURES)
                c_four_hour_TESTBTC = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_30MINUTE, '146 HOUR ago UTC ',
                                                           klines_type=HistoricalKlinesType.FUTURES)

            # Convert the fetched data to numpy arrays
            c_four_hour_TEST = np.array(c_four_hour_TEST)
            c_four_hour_TESTBTC = np.array(c_four_hour_TESTBTC)

            # Prepare the test data
            X_test, y_test, new1, mayeb1, crnew1 = prepare_test(c_four_hour_TEST, c_four_hour_TESTBTC, sc1, sc2, sc3,
                                                                sc4, sc5,
                                                                sc6, sc7, sc8, sc9, sc10, sc11, sc12, sc13, sc14, sc15,
                                                                sc16,
                                                                sc17, sc18, sc19, sc20, sc21, sc22, sc23, sc24, sc25,
                                                                sc26,
                                                                sc27, sc28, sc29, sc30, sc31, sc32, sc33, sc34, sc35,
                                                                sc36,
                                                                sc37, sc38, sc39)
            # Return the test data
            return X_test, c_four_hour_TEST, y_test


# This function calculates the percentage movement between an opening and closing value.
# It takes two arguments: 'open' and 'close', which represent the opening and closing values respectively.
def percentagemovement(open, close):
    # Convert the opening value to a float and round it to 3 decimal places
    open = round(float(open), 3)

    # Convert the closing value to a float and round it to 3 decimal places
    close = round(float(close), 3)

    # Calculate the absolute difference between the opening and closing values, divide by the opening value, and multiply by 100 to get the percentage movement
    pm = (abs(open - close) / open) * 100

    # Round the percentage movement to 2 decimal places
    pm = round(pm, 2)

    # Return the percentage movement
    return pm

# This function takes a model, test data, and test labels as input.
# It uses the model to make predictions on the test data, calculates the percentage movement for each prediction,
# and computes the final profit or loss after subtracting the commission.
# The function returns the final profit or loss.
def find_finalpo(model, xtest, ct, y_test):
    # Load the model from the given path
    model = load_model(model)

    # Use the model to make predictions on the test data
    predict = model.predict(xtest)

    # Initialize an empty list to store the binary predictions
    pred3 = []

    # For each prediction and corresponding true label
    for x, y in zip(predict, y_test):
        # If the prediction is less than or equal to 0.5, append 0 to the list
        if x <= 0.5:
            pred3.append(0)
        # If the prediction is greater than 0.5, append 1 to the list
        elif x > 0.5:
            pred3.append(1)
        # Otherwise, append -2 to the list
        else:
            pred3.append(-2)

    # Convert the list of binary predictions to a numpy array and reshape it into a column vector
    pred3 = np.array(pred3).reshape(-1, 1)

    # Initialize a variable to store the number of changes in prediction
    sume = 0

    # Initialize a variable to store the previous prediction
    y = 999

    # For each prediction
    for x in pred3:
        # If the prediction has changed from the previous prediction or if the prediction is -2, increment the count of changes
        if (x == 1 and y == 0) or (x == 0 and y == 1) or (x == 0 and y == -2) or (x == 1 and y == -2):
            sume = sume + 1
        # Update the previous prediction
        y = x

    # Calculate the commission based on the number of changes
    com = sume * 0.07

    # Initialize an empty list to store the percentage movements
    per = []

    # For each opening and closing value in the test data
    for x, y in zip(ct[:, 1:2], ct[:, 4:5]):
        # Calculate the percentage movement and append it to the list
        per.append(percentagemovement(float(x), float(y)))

    # Convert the list of percentage movements to a numpy array and reshape it into a column vector
    per = np.array(per).reshape(-1, 1)

    # Get the percentage movements for the test data
    pern = per[290:, :]

    # Initialize a variable to store the total profit or loss
    collu = 0

    # For each prediction, true label, and percentage movement
    for x, y, z in zip(pred3, y_test, pern):
        # If the prediction is correct and is not -2, add the percentage movement to the total profit or loss
        if (float(x) == float(y)) and (float(x) != -2):
            collu = collu + z
        # If the prediction is incorrect and is not -2, subtract the percentage movement from the total profit or loss
        elif (float(x) != float(y)) and (float(x) != -2):
            collu = collu - z

    # Subtract the commission from the total profit or loss to get the final profit or loss
    finalpo = collu - com

    # Return the final profit or loss
    return finalpo
