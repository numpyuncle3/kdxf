#数据分析，特征工程等
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#模型选择，辅助
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

#建模
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor


import lightgbm as lgb
import xgboost as xgb
import catboost as cb

#数据读取
train_data = pd.read_csv('eda_files/train_data_eda_8.csv')
test_data = pd.read_csv('eda_files/test_data_eda_8.csv')
pred_data = pd.read_csv('original_data/sss.csv')
pred_out_temp = test_data['outtemp']

#训练数据，测试数据的构建
#train_x_是留出法的数据，最后并没有再用
train_ = train_data[0:18832]
train_y_ = train_['gaptemp']
del train_['gaptemp']
train_x_ = train_.values

test_ = train_data[18832:len(train_data)-1]
test_y_ = test_['gaptemp']
del test_['gaptemp']
test_x_ = test_.values

train_x_spl_ = train_x_[0:15620]
train_y_spl_ = train_y_[0:15620]
test_x_spl_ = train_x_[15620:len(train_x_)]
test_y_spl_ = train_y_[15620:len(train_x_)]

train_y = train_data['gaptemp']
del train_data['gaptemp']
train_x = train_data.values
test_x = test_data.values


#ridge模型
model_ridge = Ridge(alpha = 1.48,normalize=True,random_state=2020)
model_ridge.fit(train_x,train_y)
result_ridge = model_ridge.predict(test_x)

#xgboost模型
xgb_params_final = {'eta': 0.01, 
                    'n_estimators': 445, 
                    'gamma': 0, 
                    'max_depth': 4, 
                    'min_child_weight':5,
                    'gamma':0.49,
                    'subsample': 0.76,
                    'colsample_bytree': 0.59,
                    'reg_lambda': 59,
                    'reg_alpha': 0, 
                    'colsample_bylevel': 1,
                    'seed': 2020}
model_xgb = xgb.XGBRegressor(**xgb_params_final)
model_xgb.fit(train_x,train_y)
result_xgb= model_xgb.predict(test_x)

#lightgbm模型
#训练时忘了设置随机数种子，蛋疼
model_lgb = lgb.LGBMRegressor(objective='regression',
                              metric='mse',
                              learning_rate=0.02,
                              n_estimators=6102,
                              max_depth=7,
                              num_leaves=47,
                              min_child_samples = 20,
                              min_child_weight = 0.001,
                              bagging_fraction = 0.75,
                              feature_fraction = 0.65,
                              bagging_frequency = 7,
                              lambda_l1 = 0.5,
                              lambda_l2 = 1.0
                             )
model_lgb.fit(train_x,train_y)
result_lgb= model_lgb.predict(test_x)
#catboost模型
#当时已经是最后一天了，catboost没有时间调参了。
model_ctb =  ctb.CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=8,
                             eval_metric='RMSE',
                             
                             bagging_temperature = 0.2,
                             
                             metric_period = 50,
                             random_seed = 2020,)
model_ctb.fit(train_x,train_y)
model_ctb.fit(train_x,train_y)
result_ctb= model_ctb.predict(test_x)

#最终的融合结果
result_blending = result_ctb*0.7 + result_lgb*0.2 + result_xgb*0.1
pred_data['tempreture'] = result_blending
pred_data.to_csv('result.csv', index = False)