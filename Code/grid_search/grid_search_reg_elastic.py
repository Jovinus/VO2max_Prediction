# %% Import package to use
import numpy as np
from numpy.lib.function_base import disp
import seaborn as sns 
import datatable
import pandas as pd
from my_module import *
import matplotlib.pyplot as plt
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
df_selected = df_init[(df_init['visit_num'] == 1) & (df_init['sex'] == 1)].reset_index(drop=True)
# df_selected = df_init[(df_init['visit_num'] == 1)].reset_index(drop=True)
print("Number of eq case = {}".format(len(df_selected)))
display(df_selected.head())

# %% Tran-test split for validation
from sklearn.model_selection import train_test_split

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(df_selected.drop(columns=['CRF']),
                                                    df_selected['CRF'], random_state=1004, stratify=df_selected['sex'],
                                                    test_size=0.2)
print("Train set size = {}".format(len(X_train_data)))
print("Test set size = {}".format(len(X_test_data)))

# %% Gridsearch to find best models
"""
-------------------------------- Build model -------------------------------------
There is two types of model that estimate VO2max(CRF)
- BMI
- VO2max

Adjusted with age, sex, rest_HR, MVPA
-----------------------------------------------------------------------------------
"""
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error

# feature_mask = ['AGE', 'sex', 'BMI', 'rest_HR', 'MVPA']
feature_mask = ['AGE', 'BMI', 'rest_HR', 'MVPA']

hyper_param_alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5]
hyper_param_l1_ratio = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
hyper_param_normalize = [True, False]
hyper_param_selection = ['cyclic', 'random']


results = {}

for hyper_normalize in tqdm(hyper_param_normalize, desc= 'normalize'):
    
    for hyper_alpha in tqdm(hyper_param_alpha, desc= 'alpha'):
        
        for hyper_l1_ratio in tqdm(hyper_param_l1_ratio, desc= 'l1_ratio'):
            
            for hyper_selection in tqdm(hyper_param_selection, desc= 'selection'):

                skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

                scores = []
                mse_loss = []

                for train_index, validation_index in skf.split(X_train_data, X_train_data['sex']):
                    
                        X_train = X_train_data.iloc[train_index][feature_mask]
                        X_validation = X_train_data.iloc[validation_index][feature_mask]

                        model_elastic = ElasticNet(alpha= hyper_alpha, 
                                            normalize=hyper_normalize,
                                            l1_ratio= hyper_l1_ratio,
                                            selection= hyper_selection)

                        model_elastic.fit(X=X_train_data.iloc[train_index][feature_mask], y=y_train_data.iloc[train_index])
                        
                        r2_score = model_elastic.score(X_train_data.iloc[validation_index][feature_mask], y_train_data.iloc[validation_index])
                        loss = mean_squared_error(y_true=y_train_data.iloc[validation_index], y_pred=model_elastic.predict(X_train_data.iloc[validation_index][feature_mask]))
                        
                        scores.append(r2_score)
                        mse_loss.append(loss)

                result = {'normalize': hyper_normalize, 
                          'alpha': hyper_alpha, 
                          'l1_ratio': hyper_l1_ratio, 
                          'selection': hyper_selection, 
                          'scores':scores, 
                          'mean_score':np.mean(scores), 
                          'std_score':np.std(scores),
                          'mse_loss':mse_loss, 
                          'mean_mse':np.mean(mse_loss),
                          'std_loss':np.std(mse_loss)}
                # print(result['mean_score'])
                results["normalize_" + str(hyper_normalize) + "_alpha_" + str(hyper_alpha) + "_l1_ratio_" + str(hyper_l1_ratio) + "_selection_" + str(hyper_selection)] = result

# %%
import pickle
with open('./F_results_5_elastic_reg.pickle', 'wb') as file_nm:
    pickle.dump(results, file_nm, protocol=pickle.HIGHEST_PROTOCOL)