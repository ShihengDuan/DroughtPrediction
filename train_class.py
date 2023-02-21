import numpy as np
import pandas as pd
import json
import os
from tqdm.auto import tqdm
from datetime import datetime
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from utils import interpolate_nans, date_encode, loadXY
from autogluon.tabular import TabularDataset, TabularPredictor
from multi_auto import MultilabelPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

PATH = 'DUAN-Models/classification/'
train = pd.read_csv('train_timeseries/train_timeseries.csv').set_index(['fips', 'date'])
print(train.head())
predictors = ['PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS',
              'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']

# Normalization
median = train[predictors].median()
scale = train[predictors].quantile(.75)-train[predictors].quantile(.25)
train_norm = train
train_norm[predictors] = (train[predictors]-median)/scale
train_norm['score'] = train_norm['score'].round()
soil_df = pd.read_csv('soil_data.csv')
cols = list(soil_df.columns)
variables = cols[1:]
soil_df[variables] = (soil_df[variables]-soil_df[variables].mean())/(soil_df[variables].std())

X_static_train, X_time_train, y_target_train = loadXY(train_norm)

static_data_cols = sorted(
        [c for c in soil_df.columns if c not in ["soil", "lat", "lon"]]
    )
out_label = ['SPI_1', 'SPI_2', 'SPI_3', 'SPI_4', 'SPI_5', 'SPI_6']
time_features = [str(i) for i in range(180*21)]
names = time_features+ static_data_cols + out_label

all_data = np.concatenate((X_static_train, X_time_train.reshape(X_time_train.shape[0], -1), y_target_train), axis=1)
name = []
train_data = pd.DataFrame(data=all_data, columns=names)
train_ds = TabularDataset(train_data)
# AutoGluon
predictor = MultilabelPredictor(labels=out_label, problem_types=['multiclass']*6, eval_metrics=['f1_macro']*6, path=PATH)
predictor.fit(train_data.loc[:], time_limit=700, holdout_frac=0.4, 
                                                excluded_model_types=['KNN',],
                                                # presets='best_quality,'
                                                # ag_args_fit={'num_gpus': 1},
                                               )
# Validation
print('Validation:')
val = pd.read_csv('validation_timeseries/validation_timeseries.csv').set_index(['fips', 'date'])
val_norm = val
val_norm[predictors] = (val_norm[predictors]-median)/scale
val_norm['score'] = val_norm['score'].round()
X_static_val, X_time_val, y_target_val = loadXY(val_norm)
val_data = np.concatenate((X_static_val, X_time_val.reshape(X_time_val.shape[0], -1), y_target_val), axis=1)
val_data = pd.DataFrame(data=val_data, columns=names)
val_ds = TabularDataset(val_data)
pred = predictor.predict(val_ds)

pred.to_csv(PATH+'val_pred.csv', encoding='utf-8', index=False)
val_data.to_csv(PATH+'val_data.csv', encoding='utf-8', index=False)

print('Validation F1')
print(f1_score(val_data['SPI_1'].round(), pred['SPI_1'].round(), average='macro'))
print(f1_score(val_data['SPI_2'].round(), pred['SPI_2'].round(), average='macro'))
print(f1_score(val_data['SPI_3'].round(), pred['SPI_3'].round(), average='macro'))
print(f1_score(val_data['SPI_4'].round(), pred['SPI_4'].round(), average='macro'))
print(f1_score(val_data['SPI_5'].round(), pred['SPI_5'].round(), average='macro'))
print(f1_score(val_data['SPI_6'].round(), pred['SPI_6'].round(), average='macro'))
print('Validation F1-Macro')
pred_all = np.array([pred['SPI_1'].round(), pred['SPI_2'].round(), pred['SPI_3'].round(),\
                     pred['SPI_4'].round(), pred['SPI_5'].round(), pred['SPI_6'].round()]).flatten()
true_all = np.array([val_data['SPI_1'].round(), val_data['SPI_2'].round(), val_data['SPI_3'].round(),\
                     val_data['SPI_4'].round(), val_data['SPI_5'].round(), val_data['SPI_6'].round()]).flatten()
print(f1_score(true_all, pred_all, average='macro'))

# Test

test = pd.read_csv('test_timeseries/test_timeseries.csv').set_index(['fips', 'date'])
test_norm = test
test_norm[predictors] = (test_norm[predictors]-median)/scale
test_norm['score'] = test_norm['score'].round()
X_static_test, X_time_test, y_target_test = loadXY(test_norm)
test_data = np.concatenate((X_static_test, X_time_test.reshape(X_time_test.shape[0], -1), y_target_test), axis=1)
test_data = pd.DataFrame(data=test_data, columns=names)
test_ds = TabularDataset(test_data)
pred = predictor.predict(test_ds)
print('TEST MAE')
print(mean_absolute_error(pred['SPI_1'], test_data['SPI_1']))
print(mean_absolute_error(pred['SPI_2'], test_data['SPI_2']))
print(mean_absolute_error(pred['SPI_3'], test_data['SPI_3']))
print(mean_absolute_error(pred['SPI_4'], test_data['SPI_4']))
print(mean_absolute_error(pred['SPI_5'], test_data['SPI_5']))
print(mean_absolute_error(pred['SPI_6'], test_data['SPI_6']))
print('TEST F1')
print(f1_score(test_data['SPI_1'].round(), pred['SPI_1'].round(), average='macro'))
print(f1_score(test_data['SPI_2'].round(), pred['SPI_2'].round(), average='macro'))
print(f1_score(test_data['SPI_3'].round(), pred['SPI_3'].round(), average='macro'))
print(f1_score(test_data['SPI_4'].round(), pred['SPI_4'].round(), average='macro'))
print(f1_score(test_data['SPI_5'].round(), pred['SPI_5'].round(), average='macro'))
print(f1_score(test_data['SPI_6'].round(), pred['SPI_6'].round(), average='macro'))
print('TEST F1-Macro')
pred_all = np.array([pred['SPI_1'].round(), pred['SPI_2'].round(), pred['SPI_3'].round(),\
                     pred['SPI_4'].round(), pred['SPI_5'].round(), pred['SPI_6'].round()]).flatten()
true_all = np.array([test_data['SPI_1'].round(), test_data['SPI_2'].round(), test_data['SPI_3'].round(),\
                     test_data['SPI_4'].round(), test_data['SPI_5'].round(), test_data['SPI_6'].round()]).flatten()
print(f1_score(true_all, pred_all, average='macro'))
pred.to_csv(PATH+'test_pred.csv', encoding='utf-8', index=False)
test_data.to_csv(PATH+'test_data.csv', encoding='utf-8', index=False)
