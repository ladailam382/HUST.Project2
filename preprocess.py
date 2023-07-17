import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def relative_strength_idx(df, n=14):
    close = df['close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi



def relative_strength_idx(df, n=14):
    close = df['close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi



def get_technical_indicators(dataset):
    n = dataset.shape[0]//500 + 1
    dataset = dataset.loc[::n]
    print(len(dataset))
    # Create 7 and 21 days Moving Average
    dataset['EMA_9'] = dataset['close'].ewm(9).mean().shift()
    dataset['SMA_5'] = dataset['close'].rolling(5).mean().shift()
    dataset['SMA_10'] = dataset['close'].rolling(10).mean().shift()
    dataset['SMA_15'] = dataset['close'].rolling(15).mean().shift()
    dataset['SMA_30'] = dataset['close'].rolling(30).mean().shift()

    dataset['RSI'] = relative_strength_idx(dataset).fillna(0)

    EMA_12 = pd.Series(dataset['close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(dataset['close'].ewm(span=26, min_periods=26).mean())
    dataset['MACD'] = pd.Series(EMA_12 - EMA_26)
    dataset['MACD_signal'] = pd.Series(dataset.MACD.ewm(span=9, min_periods=9).mean())

    dataset['close'] = dataset['close'].shift(-1)

    dataset = dataset.iloc[33:] # Because of moving averages and MACD line
    dataset = dataset[:-1]      # Because of shifting close price
    # dataset['date'] = dataset.index
    dataset.index = range(len(dataset))

    # dataset.drop(['open',	'high',	'low',	'adjclose',	'volume',	'ticker'], axis=1, inplace=True)
    
    final = dataset[['date']].copy()
    

    final['value'] = dataset['close']
    final['type'] = "Close"

    for i in ['EMA_9',	'SMA_5',	'SMA_10',	'SMA_15',	'SMA_30',	'RSI',	'MACD',	'MACD_signal']:
        temp = dataset[['date']].copy().loc[::n]
        temp['value'] = dataset[i]
        temp['type'] = i
        final = pd.concat([final, temp])
    print(final.tail())
    return final




def preprocess_data(test_data, is_rnn=False):
    # Split the data into x_train and y_train data sets
    x_test = []

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_test= np.array(x_test)

    if is_rnn:
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test

def preprocess_rnn_data(data):
    X_test_rnn = np.array([data[-252:]])

    return X_test_rnn