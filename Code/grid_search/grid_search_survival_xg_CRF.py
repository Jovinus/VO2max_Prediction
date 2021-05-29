# %% import package to use
import datatable
import pandas as pd
import xgboost as xgb
from IPython.display import display
import numpy as np
from xgbse.metrics import concordance_index
import os
pd.set_option('display.max_columns', None)
# %%
DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Results"
df_orig = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_for_surv.csv'), na_strings=['', 'NA']).to_pandas()
df_orig['lower_bound'] = df_orig['delta_time']
df_orig['upper_bound'] = np.where(df_orig['death'] == 1, df_orig['lower_bound'], +np.inf)
df_orig.head()

# %%
feature_names = ['AGE', 'sex', 'Smoke', 'ALC', 'CRP', 'TG', 'Diabetes', 'Hyperlipidemia', 'MBP', 'ABRP_CRF']

"""
Option 1: Adjust age
Option 2: Adjust sex
Option 3: Adjust age and sex
Option 3: Non-adjust age and sex
"""

# %%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_orig, train_size=0.8, stratify=df_orig['death'], random_state=1005)

from tqdm import tqdm
from lifelines.utils import concordance_index
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import pickle

hyper_param_depth = [3, 4, 5]
hyper_param_lr = [0.01, 0.001, 0.0001]
hyper_param_labmda = [1, 2, 3]
hyper_param_gamma = [0, 0.1, 0.2, 0.3]


results = {}

for hyper_lr in tqdm(hyper_param_lr, desc= 'l_rate'):

    for hyper_depth in tqdm(hyper_param_depth, desc= 'depth'):
        
        for hyper_labmda in tqdm(hyper_param_labmda, desc= 'lambda'):
            
            for hyper_gamma in tqdm(hyper_param_gamma, desc= 'gamma'):

                skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

                scores = []

                for train_index, validation_index in skf.split(train_set, train_set['death']):
                        X_train = train_set.iloc[train_index][feature_names].values
                        dtrain = xgb.DMatrix(X_train)
                        dtrain.set_float_info('label_lower_bound', train_set.iloc[train_index]['lower_bound'].values)
                        dtrain.set_float_info('label_upper_bound', train_set.iloc[train_index]['upper_bound'].values)

                        X_validation = train_set.iloc[validation_index][feature_names].values
                        dvalidation = xgb.DMatrix(X_validation)
                        dvalidation.set_float_info('label_lower_bound', train_set.iloc[validation_index]['lower_bound'].values)
                        dvalidation.set_float_info('label_upper_bound', train_set.iloc[validation_index]['upper_bound'].values)

                        params = {'objective': 'survival:aft',
                            'eval_metric': 'aft-nloglik',
                            'aft_loss_distribution': 'normal',
                            'aft_loss_distribution_scale': 1.20,
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
                        
                        c_index = concordance_index(train_set.iloc[validation_index]['lower_bound'].values, 
                                    model_xgb.predict(dvalidation), 
                                    event_observed=train_set.iloc[validation_index]['death'].astype(int))

                        scores.append(c_index)

                result = {'max_depth': hyper_depth, 'learning_rate': hyper_lr, 'lambda':hyper_labmda, 'gamma': hyper_gamma, 'scores':scores, 'mean_score':np.mean(scores), 'std_score':np.std(scores)}
                # print(result['mean_score'])
                results["depth_" + str(hyper_depth) + "_lr_" + str(hyper_lr) + "_lambda_" + str(hyper_labmda) + "_gamma_" + str(hyper_gamma)] = result
# %%
with open('../../Model/MF_10_survival_xg_results_CRF_var.pickle', 'wb') as file_nm:
    pickle.dump(results, file_nm, protocol=pickle.HIGHEST_PROTOCOL)