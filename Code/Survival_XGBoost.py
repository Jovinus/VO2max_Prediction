# %% import package to use
import datatable
import pandas as pd
import xgboost as xgb
from IPython.display import display
import numpy as np
pd.set_option('display.max_columns', None)
# %%
DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Results"
df_orig = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_for_surv.csv'), 
                          na_strings=['', 'NA']).to_pandas()

# %% Data preprocessing for Survival XGBoost

df_orig['lower_bound'] = df_orig['delta_time']
df_orig['upper_bound'] = np.where(df_orig['death'] == 1, df_orig['lower_bound'], np.inf)
df_orig.head()
# %%
feature_names = ['Smoke', 'ALC', 'CRP', 'TG', 'Diabetes', 'Hypertension', 
                 'Hyperlipidemia', 'HDL_C', 'LDL_C', 'MBP', 'ABRP_CRF']

'age', 'sex' "3가지 조건으로"

# %%
from sklearn.model_selection import train_test_split