import keras
import catboost as cb
import lightgbm as lgbm
import xgboost as xgb

from tensorflow.keras import backend as K

from config import *

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 



model_lstm = keras.models.load_model('weights/lstm.h5', custom_objects={'root_mean_squared_error' : root_mean_squared_error})

model_gru = keras.models.load_model('weights/gru.h5', custom_objects={'root_mean_squared_error' : root_mean_squared_error})

model_catboost = cb.CatBoostRegressor()
model_catboost.load_model('weights/catboost.cbm')


model_xgboost = xgb.Booster()
model_xgboost.load_model('weights/xgb.json')


# model_lgbm = LGBMRegressor(learning_rate        = LGBM_CONFIG["learning_rate"],
#                         n_estimators         = LGBM_CONFIG["n_estimators"],
#                         max_depth            = LGBM_CONFIG["max_depth"],
#                         colsample_bytree     = LGBM_CONFIG["colsample_bytree"],
#                         subsample            = LGBM_CONFIG["subsample"],
#                         min_child_samples    = LGBM_CONFIG["min_child_samples"],
#                         eval_metric='rmse')

model_lgbm = lgbm.Booster(model_file='weights/lgbm.txt')

model_map = {
        "Catboost": model_catboost,
        "LightGBM": model_lgbm,
        "XGBoost": model_xgboost,
        "LSTM": model_lstm,
        "GRU": model_gru
}
