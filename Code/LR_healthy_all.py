# %% Import package to use
import numpy as np
from numpy.lib.function_base import disp
import seaborn as sns 
import datatable
import pandas as pd
from my_module import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from IPython.display import display
pd.set_option("display.max_columns", None)
import os

# %% Load dataset
DATA_PATH = "/Users/lkh256/Studio/VO2max_Prediction/Data"
df_init = datatable.fread(os.path.join(DATA_PATH, 'healthy_eq.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_surv = datatable.fread(os.path.join(DATA_PATH, 'healthy_survival.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_init['SM_DATE'] = df_init['SM_DATE'].astype('datetime64')
df_surv['SM_DATE'] = df_surv['SM_DATE'].astype('datetime64')
print("Number of samples = {}".format(len(df_init)))
display(df_init.head())

# %% Check missings
print("Check their is any missing variables in dataset: \n", df_init.isnull().sum())

# %% Sort visit number and select rows to analysis
df_init['visit_num'] = df_init.groupby(['HPCID'])['SM_DATE'].apply(pd.Series.rank)
df_surv['visit_num'] = df_surv.groupby(['HPCID'])['SM_DATE'].apply(pd.Series.rank)
df_selected = df_init[df_init['visit_num'] == 1].reset_index(drop=True)
df_surv = df_surv[df_surv['visit_num'] == 1].reset_index(drop=True)
print("Number of case = {}".format(len(df_selected)))
display(df_selected.head(), df_surv.head())

# %% Demographic statics
from tableone import TableOne
columns = ['sex', 'AGE', 'percentage_fat', 'BMI', 'MVPA', 'rest_HR', 'VO2max', 'CRF', 'ASMI', 'Smoke', 'death']

categorical = ['sex', 'MVPA', 'Smoke', 'death']

group_by = 'sex'

dem_table = TableOne(data=df_selected, columns=columns, categorical=categorical, groupby=group_by, pval=True)

display(dem_table)
dem_table.to_excel("../Results/MF_demo_stats.xlsx")

# %% Variable correlation
plt.figure(figsize=(12, 10))
sns.heatmap(df_selected[columns].corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues', annot_kws={'size':15})
plt.show()

# %% Tran-test split for validation
from sklearn.model_selection import train_test_split
#### Male
X_train, X_test, y_train, y_test = train_test_split(df_selected.drop(columns=['VO2max']),
                                                    df_selected['VO2max'], random_state=1004,
                                                    test_size=0.2)
print("Train set size = {}".format(len(X_train)))
print("Test set size = {}".format(len(X_test)))

# %% Build model and make estimate to percentiles
"""
-------------------------------- Build model -------------------------------------
There is two types of model that estimate VO2max(CRF)
- BMI
- VO2max

Adjusted with age, sex, rest_HR, MVPA
-----------------------------------------------------------------------------------
"""
df_selected, df_surv = make_model_tertile(df_selected, df_surv, X_train, y_train, X_test, y_test)
display(df_selected.head(), df_surv.head())
# %% Save results

df_selected.to_csv('../Results/MF_health_for_surv.csv', index=False, encoding='utf-8-sig')
df_surv.to_csv('../Results/MF_health_all_for_surv.csv', index=False, encoding='utf-8-sig')
