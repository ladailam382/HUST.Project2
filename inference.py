from yahoo_fin.stock_info import get_data
import datetime
from preprocess import preprocess_data, preprocess_rnn_data
from model import model_map
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import xgboost as xgb


def get_update_data(start = "01/01/2000", end = "01/07/2023"):
    prev_day = datetime.date.today() - datetime.timedelta(days=1)
    prev_day = prev_day.strftime("%d/%m/%Y")

    if end > prev_day:
        end = prev_day

    # print(start, end)

    infy_df = get_data("INFY.NS", start_date=start, end_date=end, index_as_date=True, interval="1d")

    data = infy_df[["close"]]
    data['date'] = data.index
    data.reset_index(drop=True, inplace=True)
    return data[-5000:]

def predict_price(modeltype, data, interval=1):

    # org_data = data.copy()
    # print(data.iloc[-1, 1])
    # end_date = datetime.datetime.strptime(str(data.iloc[-1, 1]), "%d/%m/%y %H:%M:%S")
    while interval > 0:
        end_date = data.iloc[-1, 1]

        model = model_map[modeltype]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_test_data = scaler.fit_transform(data[['close']].copy())

        if modeltype in ['LSTM', 'GRU']:
            preprocessed_data = preprocess_data(scaled_test_data, is_rnn=True)
        else:
            preprocessed_data = preprocess_data(scaled_test_data)

        df2 = pd.DataFrame({"date":[end_date + datetime.timedelta(days=1)]})
        
        data = pd.concat([data, df2], ignore_index = True)

        if modeltype == "XGBoost":
            preprocessed_data = xgb.DMatrix(preprocessed_data)
        # else:
        predict = model.predict(preprocessed_data)
        data.iloc[-1:, 0] = predict[-1:]/scaler.scale_[0]

        interval -= 1
    # print(data.tail())
    return data[-100:]



