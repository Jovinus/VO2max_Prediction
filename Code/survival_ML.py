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
import kaplanmeier as km

# %%
DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Results"
df_orig = datatable.fread(os.path.join(DATA_PATH, 'MF_general_all_for_surv.csv'), 
                          encoding='utf-8-sig', 
                          na_strings=['', 'NA']).to_pandas()
df_orig.head()

# %%
feature_names = ['AGE', 'sex', 'rest_HR', 'Smoke', 'ALC', 'CRP', 'CHOLESTEROL', 
                 'TG', 'Diabetes', 'Hypertension', 'Hyperlipidemia', 'Hepatatis', 
                 'HDL_C', 'LDL_C', 'MBP', 'ABRP_CRF']

X_data = df_orig[feature_names]
X_data = X_data.fillna(X_data.median()).values.astype(float)

y_data = np.zeros(df_orig.shape[0], dtype={'names':('death', 'delta_time'), 
                                           'formats':(bool, float)})
y_data['death'] = df_orig['death'].astype(bool)
y_data['delta_time'] = df_orig['delta_time'].astype(float)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                    train_size=0.7, stratify=y_data['death'])

# %%
from sksurv.ensemble import RandomSurvivalForest

surv_rf = RandomSurvivalForest(n_estimators=100, max_depth=6, n_jobs=-1)
surv_rf.fit(X_train, y_train)
print("Train Concordence Index = {}".format(surv_rf.score(X_train, y_train)))
print("Test Concordence Index = {}".format(surv_rf.score(X_test, y_test)))

# %%
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(surv_rf, n_iter=15, random_state=1004)
perm.fit(X_test, y_test)
eli5.show_weights(perm, feature_names=feature_names)
# %%

# %%
# PySurvival