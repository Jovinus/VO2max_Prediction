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
DATA_PATH = "/Users/lkh256/Studio/VO2max_Prediction/Results"
df_health = datatable.fread(os.path.join(DATA_PATH, 'MF_health_eq_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_general = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()

df_health_m = datatable.fread(os.path.join(DATA_PATH, 'M_health_eq_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_general_m = datatable.fread(os.path.join(DATA_PATH, 'M_general_eq_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()

df_health_f = datatable.fread(os.path.join(DATA_PATH, 'F_health_eq_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_general_f = datatable.fread(os.path.join(DATA_PATH, 'F_general_eq_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()

df_health_all = datatable.fread(os.path.join(DATA_PATH, 'MF_health_all_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_general_all = datatable.fread(os.path.join(DATA_PATH, 'MF_general_all_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()

df_health_m_all = datatable.fread(os.path.join(DATA_PATH, 'M_health_all_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_general_m_all = datatable.fread(os.path.join(DATA_PATH, 'M_general_all_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()

df_health_f_all = datatable.fread(os.path.join(DATA_PATH, 'F_health_all_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
df_general_f_all = datatable.fread(os.path.join(DATA_PATH, 'F_general_all_for_surv.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()

# %%
print(len(df_health))
print(len(df_general))

# %%
default_column = ['HPCID', 'SM_DATE', 'AGE', 'sex', 'death', 'delta_time', 'visit_num', 'CRF', 'ABRP_VO2max', 
                  'ABRP_CRF', 'ABR_VO2max', 'ABR_CRF','ABP_VO2max', 'ABP_CRF', 'APRP_VO2max', 
                  'APRP_CRF', 'APR_VO2max', 'APR_CRF', 'APP_VO2max', 'APP_CRF', 'CRF_tertile', 
                  'CRF_tertile_nm', 'ABRP_CRF_tertile', 'ABRP_CRF_tertile_nm', 'APRP_CRF_tertile', 
                  'APRP_CRF_tertile_nm', 'CRF_qualtile', 'CRF_qualtile_nm', 'ABRP_CRF_qualtile', 
                  'ABRP_CRF_qualtile_nm', 'APRP_CRF_qualtile', 'APRP_CRF_qualtile_nm']

select_column = ['Smoke', 'ALC', 'sex', 'BMI', 'MVPA', 'Diabetes', 'Hypertension', 
                 'Hyperlipidemia', 'Hepatatis']

# 'Stroke', 'Angina', 'MI', 'Asthma', 'Cancer'

column_choosed = default_column + select_column

# %%
SAVE_PATH = "/Users/lkh256/Studio/VO2max_Prediction/Data/Survival_set"
df_health[column_choosed].to_csv(os.path.join(SAVE_PATH, 'MF_health_eq_survival.csv'), index=False, encoding='utf-8-sig')
df_general[column_choosed].to_csv(os.path.join(SAVE_PATH, 'MF_general_eq_survival.csv'), index=False, encoding='utf-8-sig')

df_health_m[column_choosed].to_csv(os.path.join(SAVE_PATH, 'M_health_eq_survival.csv'), index=False, encoding='utf-8-sig')
df_general_m[column_choosed].to_csv(os.path.join(SAVE_PATH, 'M_general_eq_survival.csv'), index=False, encoding='utf-8-sig')

df_health_f[column_choosed].to_csv(os.path.join(SAVE_PATH, 'F_health_eq_survival.csv'), index=False, encoding='utf-8-sig')
df_general_f[column_choosed].to_csv(os.path.join(SAVE_PATH, 'F_general_eq_survival.csv'), index=False, encoding='utf-8-sig')

df_health_all[column_choosed].to_csv(os.path.join(SAVE_PATH, 'MF_health_all_survival.csv'), index=False, encoding='utf-8-sig')
df_general_all[column_choosed].to_csv(os.path.join(SAVE_PATH, 'MF_general_all_survival.csv'), index=False, encoding='utf-8-sig')

df_health_m_all[column_choosed].to_csv(os.path.join(SAVE_PATH, 'M_health_all_survival.csv'), index=False, encoding='utf-8-sig')
df_general_m_all[column_choosed].to_csv(os.path.join(SAVE_PATH, 'M_general_all_survival.csv'), index=False, encoding='utf-8-sig')

df_health_f_all[column_choosed].to_csv(os.path.join(SAVE_PATH, 'F_health_all_survival.csv'), index=False, encoding='utf-8-sig')
df_general_f_all[column_choosed].to_csv(os.path.join(SAVE_PATH, 'F_general_all_survival.csv'), index=False, encoding='utf-8-sig')

# %%
