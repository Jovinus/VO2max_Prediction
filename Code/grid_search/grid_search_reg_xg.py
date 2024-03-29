# %% Import package to use
import numpy as np
from numpy.lib.function_base import disp
import datatable
import pandas as pd
import xgboost as xgb
from IPython.display import display
pd.set_option("display.max_columns", None)
import os

# %% Load dataset
DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Data"
df_init = datatable.fread(os.path.join(DATA_PATH, 'general_eq.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_init['SM_DATE'] = df_init['SM_DATE'].astype('datetime64')

df_init = df_init.fillna(df_init.median())

print("Number of samples = {}".format(len(df_init)))
display(df_init.head())

# %% Check missings
print("Check their is any missing variables in dataset: \n", df_init.isnull().sum())

# %% Sort visit number and select rows to analysis
df_init['visit_num'] = df_init.groupby(['HPCID'])['SM_DATE'].apply(pd.Series.rank)
df_selected = df_init[df_init['visit_num'] == 1].reset_index(drop=True)
print("Number of eq case = {}".format(len(df_selected)))
display(df_selected.head())

# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df_selected,  
                                       random_state=1004, 
                                       stratify=df_selected['sex'], 
                                       test_size=0.2)

print("Train set size = {}".format(len(train_set)))
print("Test set size = {}".format(len(test_set)))
# %% Cross-Val Model

from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

hyper_param_depth = [3, 4, 5]
hyper_param_lr = [0.01, 0.001, 0.0001]
hyper_param_labmda = [1, 2, 3]
hyper_param_gamma = [0, 0.1, 0.2, 0.3]

feature_mask = ['AGE', 'sex', 'BMI', 'rest_HR', 'MVPA']
results = {}

for hyper_lr in tqdm(hyper_param_lr, desc= 'l_rate'):

    for hyper_depth in tqdm(hyper_param_depth, desc= 'depth'):
        
        for hyper_labmda in tqdm(hyper_param_labmda, desc= 'lambda'):
            
            for hyper_gamma in tqdm(hyper_param_gamma, desc= 'gamma'):

                skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

                scores = []
                mse_loss = []

                for train_index, validation_index in skf.split(train_set, train_set['sex']):
                        X_train = train_set.iloc[train_index][feature_mask].values
                        y_train = train_set.iloc[train_index]['CRF'].values
                        dtrain = xgb.DMatrix(X_train, label=y_train)
                        
                        X_validation = train_set.iloc[validation_index][feature_mask].values
                        y_validation = train_set.iloc[validation_index]['CRF'].values
                        dvalidation = xgb.DMatrix(X_validation, label=y_validation)

                        params = {'objective': 'reg:squarederror',
                            'eval_metric': 'rmse',
                            'tree_method': 'gpu_hist', 
                            'gpu_id': '0',
                            'learning_rate': hyper_lr, 
                            'max_depth': hyper_depth,
                            'lambda': hyper_labmda,
                            'gamma': hyper_gamma}

                        model_xgb = xgb.train(params, dtrain, 
                                            num_boost_round=20000, 
                                            evals=[(dvalidation, 'validation')], 
                                            verbose_eval=0, 
                                            early_stopping_rounds=1000)
                        
                        score = r2_score(y_validation, model_xgb.predict(dvalidation))
                        loss = mean_squared_error(y_true=y_validation, y_pred=model_xgb.predict(dvalidation))

                        scores.append(score)
                        mse_loss.append(loss)

                result = {'max_depth': hyper_depth, 
                          'learning_rate': hyper_lr, 
                          'lambda':hyper_labmda, 
                          'gamma': hyper_gamma, 
                          'scores':scores, 
                          'mean_score':np.mean(scores), 
                          'std_score':np.std(scores), 
                          'mse_loss':mse_loss, 
                          'mean_mse':np.mean(mse_loss),
                          'std_loss':np.std(mse_loss)}
                # print(result['mean_score'])
                results["depth_" + str(hyper_depth) + "_lr_" + str(hyper_lr) + "_lambda_" + str(hyper_labmda) + "_gamma_" + str(hyper_gamma)] = result
# %%
with open('./MF_results_5_xg_reg.pickle', 'wb') as file_nm:
    pickle.dump(results, file_nm, protocol=pickle.HIGHEST_PROTOCOL)