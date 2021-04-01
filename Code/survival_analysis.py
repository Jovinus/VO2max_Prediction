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

# %%
DATA_PATH = "/Users/lkh256/Studio/VO2max_Prediction/Data/Survival_set"
df_health = datatable.fread(os.path.join(DATA_PATH, 'health_survival.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_health['ALC'] = np.where(df_health['ALC_YS'] == 1, 1, 0)
# %%
display(df_health.head())

# %%
target_var = ['death', 'delta_time']
input_focus_var = ['ABRP_VO2max']
input_adjust_var = ['Smoke', 'ALC', 'sex', 'BMI', 'Diabetes', 'Hypertension', 'Hyperlipidemia', 'Hepatatis', 'Stroke', 'Angina']
# input_adjust_var = ['Smoke', 'ALC', 'sex', 'BMI', 'MVPA', 'Diabetes', 'Hypertension', 'HTN_med', 
#              'Hyperlipidemia', 'Hepatatis', 'Stroke', 'Angina', 'MI', 'Asthma', 'Cancer']
y = df_health[target_var].to_records(index=False)
X = df_health[input_focus_var + input_adjust_var].astype(float)

# %%
from sksurv.linear_model import CoxPHSurvivalAnalysis
cox_model = CoxPHSurvivalAnalysis(alpha=0, ties="breslow", n_iter=100, tol=1e-9, verbose=0)
cox_model.fit(X, y)
# %%
from sksurv.datasets import load_whas500
X, y = load_whas500()
estimator = CoxPHSurvivalAnalysis().fit(X, y)
chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:10])
for fn in chf_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
plt.ylim(0, 1)
plt.show()
# %%
np.array(zip(df_health[target_var[0]], df_health[target_var[1]]))
# %%
df_health[target_var].to_records(index=False)
# %%
X.isnull().sum()
# %%
