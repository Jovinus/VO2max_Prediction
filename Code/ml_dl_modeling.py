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
import shap

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

# %% Demographic statics
from tableone import TableOne

columns = ['AGE', 'percentage_fat', 'BMI', 'HDL_C', 'rest_HR', 'VO2max', 'CRF', 
           'max_heart_rate', 'ASMI', 'Smoke', 'ALC', 'MVPA', 'Diabetes', 'Hypertension', 
           'Hyperlipidemia', 'Hepatatis', 'death']

categorical = ['MVPA', 'Smoke', 'death', 'ALC', 'Diabetes', 
               'Hypertension', 'Hyperlipidemia', 'Hepatatis']

group_by = 'sex'

dem_table = TableOne(data=df_selected, columns=columns, categorical=categorical, groupby=group_by, pval=True)

display(dem_table)
# dem_table.to_excel("../Results/MF_general_eq_demo_stats.xlsx")

# %% Variable correlation
plt.figure(figsize=(10, 10))
sns.heatmap(df_selected[columns].corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues', annot_kws={'size':10})
plt.show()

# %% Tran-test split for validation
from sklearn.model_selection import train_test_split

columns = ['AGE', 'sex', 'percentage_fat', 'BMI', 'HDL_C', 'rest_HR', 
           'ASMI', 'Smoke', 'ALC', 'MVPA', 'Diabetes', 'Hypertension', 
           'Hyperlipidemia', 'Hepatatis']

categorical = ['MVPA', 'Smoke', 'ALC', 'Diabetes', 
               'Hypertension', 'Hyperlipidemia', 'Hepatatis']

X_train, X_test, y_train, y_test = train_test_split(df_selected[columns],
                                                    df_selected['VO2max'], random_state=1004, stratify=df_selected['sex'],
                                                    test_size=0.3)
print("Train set size = {}".format(len(X_train)))
print("Test set size = {}".format(len(X_test)))
# %% CatBoostRegressor
from catboost import CatBoostRegressor
model = CatBoostRegressor(learning_rate=0.03, iterations=15000, task_type='GPU')
model.fit(X_train, y_train, verbose=0, cat_features=categorical)
get_metric(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_size=(10,5))
shap.summary_plot(shap_values, X_test, plot_type='bar')

### 해석할 때 Permutation 해야하는데 오래걸림
tmp = pd.DataFrame({'feature':model.feature_names_, 'importance':model.feature_importances_})
plt.figure(figsize=(10,10))
plt.title('Feature Importance of Variables', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
sns.barplot(data=tmp.sort_values('importance', ascending=False), x='importance', y='feature')
plt.show()

# %% Tran-test split for validation
from sklearn.model_selection import train_test_split

columns = ['AGE', 'sex', 'BMI', 'rest_HR', 'MVPA']

categorical = ['MVPA', 'sex']

X_train, X_test, y_train, y_test = train_test_split(df_selected[columns],
                                                    df_selected['VO2max'], random_state=1004, stratify=df_selected['sex'],
                                                    test_size=0.3)
print("Train set size = {}".format(len(X_train)))
print("Test set size = {}".format(len(X_test)))
# %% CatBoostRegressor
from catboost import CatBoostRegressor
model = CatBoostRegressor(learning_rate=0.03, iterations=1000, task_type='GPU')
model.fit(X_train, y_train, verbose=0, cat_features=categorical)
get_metric(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_size=(10,5))
shap.summary_plot(shap_values, X_test, plot_type='bar')
# %%
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

f, ax = plt.subplots(1, figsize = (8,5))
sm.graphics.mean_diff_plot(y_test, model.predict(X_test), ax = ax)

plt.show()
# %%
