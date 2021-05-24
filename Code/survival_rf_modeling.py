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
import kaplanmeier as km

# %%
DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Results"
df_orig = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_for_surv.csv'), 
                          encoding='utf-8-sig', 
                          na_strings=['', 'NA']).to_pandas()
df_orig.head()

# %%
from sklearn.model_selection import train_test_split


feature_names = ['Smoke', 'ALC', 'CRP', 'TG', 'Diabetes', 'Hypertension', 
                 'Hyperlipidemia', 'HDL_C', 'LDL_C', 'MBP', 'ABRP_CRF']

'age', 'sex' "3가지 조건으로"

train_data , test_data = train_test_split(df_orig, 
                                          train_size=0.7, 
                                          stratify=df_orig['death'], 
                                          random_state=1004)

X_train, X_test = train_data[feature_names], test_data[feature_names]
T_train, T_test = train_data['delta_time'].values, test_data['delta_time'].values
E_train, E_test = train_data['death'].values, test_data['death'].values



#%%
from pysurvival.models.survival_forest import RandomSurvivalForestModel

rsf_model = RandomSurvivalForestModel(num_trees=150)
# %%

rsf_model.fit(X_train, T_train, E_train, 
              max_features="sqrt", max_depth=5, min_node_size=20, seed=1004)

print(rsf_model.variable_importance)
# %%
from pysurvival.utils.metrics import concordance_index

#### 5 - Cross Validation / Model Performances
c_index = concordance_index(rsf_model, X_train, T_train, E_train) #0.81
print('Train C-index: {:.2f}'.format(c_index))
c_index = concordance_index(rsf_model, X_test, T_test, E_test) #0.81
print('Test C-index: {:.2f}'.format(c_index))
# %%
