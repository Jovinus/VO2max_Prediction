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
DATA_PATH = "/Users/lkh256/Studio/VO2max_Prediction/Results"
df_init = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_surv = datatable.fread(os.path.join(DATA_PATH, 'MF_general_all_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
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
# %%
plt.figure(figsize=(10, 10))
sns.histplot(x='AGE', data=df_init, kde=True)
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(x='AGE', data=df_init, kde=True, hue='sex')
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(x='CRF', data=df_init, kde=True)
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(x='CRF', data=df_init, kde=True, hue='sex')
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(x='ABR_CRF', data=df_init, kde=True)
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(x='ABR_CRF', data=df_init, kde=True, hue='sex')
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(x='APRP_CRF', data=df_init, kde=True)
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(x='APRP_CRF', data=df_init, kde=True, hue='sex')
plt.show()
# %% BMI
from statsmodels.graphics.agreement import mean_diff_plot

f, ax = plt.subplots(1, figsize=(10,10))
mean_diff_plot(df_init['ABRP_CRF'], df_init['CRF'], ax=ax)
plt.show()

f, ax = plt.subplots(1, figsize=(10,10))
mean_diff_plot(df_init['ABR_CRF'], df_init['CRF'], ax=ax)
plt.show()

f, ax = plt.subplots(1, figsize=(10,10))
mean_diff_plot(df_init['ABP_CRF'], df_init['CRF'], ax=ax)
plt.show()
# %% Percentage Fat

f, ax = plt.subplots(1, figsize=(10,10))
mean_diff_plot(df_init['APRP_CRF'], df_init['CRF'], ax=ax)
plt.show()

f, ax = plt.subplots(1, figsize=(10,10))
mean_diff_plot(df_init['APR_CRF'], df_init['CRF'], ax=ax)
plt.show()

f, ax = plt.subplots(1, figsize=(10,10))
mean_diff_plot(df_init['APP_CRF'], df_init['CRF'], ax=ax)
plt.show()

# %% BMI Equation
from statsmodels.graphics.api import qqplot
import scipy.stats as stats

f, ax = plt.subplots(1, figsize=(10,10))
qqplot((df_init['CRF'] - df_init['ABRP_CRF']), dist=stats.t, fit=True, line="45", ax=ax)
plt.show()

f, ax = plt.subplots(1, figsize=(10,10))
qqplot((df_init['CRF'] - df_init['ABR_CRF']), dist=stats.t, fit=True, line="45", ax=ax)
plt.show()

f, ax = plt.subplots(1, figsize=(10,10))
qqplot((df_init['CRF'] - df_init['ABP_CRF']), dist=stats.t, fit=True, line="45", ax=ax)
plt.show()
# %% Percentage Fat Equation
f, ax = plt.subplots(1, figsize=(10,10))
qqplot((df_init['CRF'] - df_init['APRP_CRF']), dist=stats.t, fit=True, line="45", ax=ax)
plt.show()
f, ax = plt.subplots(1, figsize=(10,10))
qqplot((df_init['CRF'] - df_init['APR_CRF']), dist=stats.t, fit=True, line="45", ax=ax)
plt.show()
f, ax = plt.subplots(1, figsize=(10,10))
qqplot((df_init['CRF'] - df_init['APP_CRF']), dist=stats.t, fit=True, line="45", ax=ax)
plt.show()
# %%
