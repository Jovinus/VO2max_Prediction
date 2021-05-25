# %% Import package to use
import numpy as np
from numpy.lib.function_base import disp
import seaborn as sns 
import datatable
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
pd.set_option("display.max_columns", None)
import os

# %% Load dataset
DATA_PATH = "/home/khl256/Studio/VO2max_Prediction/Data"
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

# %% Tran-test split for validation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_selected.drop(columns=['VO2max']),
                                                    df_selected['VO2max'], random_state=1005, stratify=df_selected['death'],
                                                    test_size=0.2)
print("Train set size = {}".format(len(X_train)))
print("Test set size = {}".format(len(X_test)))

# %% Gridsearch to find best models
"""
-------------------------------- Build model -------------------------------------
There is two types of model that estimate VO2max(CRF)
- BMI
- VO2max

Adjusted with age, sex, rest_HR, MVPA
-----------------------------------------------------------------------------------
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

feature_mask = ['AGE', 'sex', 'BMI', 'rest_HR', 'MVPA']

rf_reg = RandomForestRegressor(n_jobs=-1)
hyper_param = {'n_estimators':range(100, 1000, 100), 
               'criterion':['mse', 'mae'], 
               'min_samples_split':[2, 3, 4], 
               'max_features':['auto', 'sqrt', 'log2']}


grid_search = GridSearchCV(estimator=rf_reg, 
                           param_grid=hyper_param, 
                           scoring='r2', 
                           n_jobs=-1, 
                           cv=10)

grid_search.fit(X_train[feature_mask].astype(float), y_train)
print(grid_search.score(X_train[feature_mask].astype(float), y_train))
print(grid_search.score(X_test[feature_mask].astype(float), y_test))

# %%
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('results_rf_reg.csv', encoding='utf-8-sig')
# %%
print("Best hyperparameters : \n" ,grid_search.best_params_)
# %%
