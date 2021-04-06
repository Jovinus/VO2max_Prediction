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

df_init = df_init.fillna(df_init.median())
df_surv = df_surv.fillna(df_surv.median())

print("Number of samples = {}".format(len(df_init)))
display(df_init.head())

# %% Check missings
print("Check their is any missing variables in dataset: \n", df_init.isnull().sum())

# %% Sort visit number and select rows to analysis
df_init['visit_num'] = df_init.groupby(['HPCID'])['SM_DATE'].apply(pd.Series.rank)
df_surv['visit_num'] = df_surv.groupby(['HPCID'])['SM_DATE'].apply(pd.Series.rank)
df_selected = df_init[df_init['visit_num'] == 1].reset_index(drop=True)
df_surv = df_surv[df_surv['visit_num'] == 1].reset_index(drop=True)
print("Number of eq case = {}".format(len(df_selected)))
print("Number of surv case = {}".format(len(df_surv)))
display(df_selected.head(), df_surv.head())

# %% Demographic statics
from tableone import TableOne
columns = ['AGE', 'percentage_fat', 'BMI', 'HDL-C', 'MVPA', 'rest_HR', 'VO2max', 'CRF', 'max_heart_rate', 'ASMI', 'Smoke', 'ALC', 'death']

categorical = ['MVPA', 'Smoke', 'death', 'ALC']

group_by = 'sex'

dem_table = TableOne(data=df_selected, columns=columns, categorical=categorical, groupby=group_by, pval=True)

display(dem_table)
dem_table.to_excel("../Results/MF_health_eq_demo_stats.xlsx")

dem_table = TableOne(data=df_surv, columns=columns, categorical=categorical, groupby=group_by, pval=True)

display(dem_table)
dem_table.to_excel("../Results/MF_health_all_demo_stats.xlsx")

# %% Variable correlation
plt.figure(figsize=(12, 10))
sns.heatmap(df_selected[columns].corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues', annot_kws={'size':15})
plt.show()

# %% Tran-test split for validation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_selected.drop(columns=['VO2max']),
                                                    df_selected['VO2max'], random_state=1004, stratify=df_selected['sex'],
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

df_selected.to_csv('../Results/MF_health_eq_for_surv.csv', index=False, encoding='utf-8-sig')
df_surv.to_csv('../Results/MF_health_all_for_surv.csv', index=False, encoding='utf-8-sig')

# %% Tran-test split for validation - Male
df_selected_m = df_selected[df_selected['sex'] == 0]
df_surv_m = df_surv[df_surv['sex'] == 0]

X_train, X_test, y_train, y_test = train_test_split(df_selected_m.drop(columns=['VO2max']),
                                                    df_selected_m['VO2max'], random_state=1004,
                                                    test_size=0.2)
print("Train set size = {}".format(len(X_train)))
print("Test set size = {}".format(len(X_test)))

# %% Build model and make estimate to percentiles - Male
"""
-------------------------------- Build model -------------------------------------
There is two types of model that estimate VO2max(CRF)
- BMI
- VO2max

Adjusted with age, rest_HR, MVPA
-----------------------------------------------------------------------------------
"""
df_selected_m, df_surv_m = make_model_tertile(df_selected_m, df_surv_m, X_train, y_train, X_test, y_test)
display(df_selected_m.head(), df_surv_m.head())
# %% Save results

df_selected_m.to_csv('../Results/M_health_eq_for_surv.csv', index=False, encoding='utf-8-sig')
df_surv_m.to_csv('../Results/M_health_all_for_surv.csv', index=False, encoding='utf-8-sig')

# %% Tran-test split for validation - Female
df_selected_f = df_selected[df_selected['sex'] == 1]
df_surv_f = df_surv[df_surv['sex'] == 1]

X_train, X_test, y_train, y_test = train_test_split(df_selected_f.drop(columns=['VO2max']),
                                                    df_selected_f['VO2max'], random_state=1004,
                                                    test_size=0.2)
print("Train set size = {}".format(len(X_train)))
print("Test set size = {}".format(len(X_test)))

# %% Build model and make estimate to percentiles - Female
"""
-------------------------------- Build model -------------------------------------
There is two types of model that estimate VO2max(CRF)
- BMI
- VO2max

Adjusted with age, rest_HR, MVPA
-----------------------------------------------------------------------------------
"""
df_selected_f, df_surv_f = make_model_tertile(df_selected_f, df_surv_f, X_train, y_train, X_test, y_test)
display(df_selected_f.head(), df_surv_f.head())
# %% Save results

df_selected_f.to_csv('../Results/F_health_eq_for_surv.csv', index=False, encoding='utf-8-sig')
df_surv_f.to_csv('../Results/F_health_all_for_surv.csv', index=False, encoding='utf-8-sig')
# %%
