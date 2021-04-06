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
DATA_PATH = '/Users/lkh256/Studio/VO2max_Prediction/Data/Survival_set'
df_init = datatable.fread(os.path.join(DATA_PATH, 'MF_general_all_survival.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
display(df_init.head())
# %%
#### make input variable for kaplanmeier curve
time_event = df_init['delta_time'].astype(int)
censoring = df_init['death'].astype(int)
labx = df_init['CRF_tertile']

out = km.fit(time_event, censoring, labx)

km.plot(out, cmap='Set2', cii_lines='line', cii_alpha=0.05)
plt.show()

# %% 
from lifelines import datasets, CoxPHFitter
from patsy import dmatrices

df_init[['delta_time', 'death', 'sex', 'Smoke', 'ALC', 'MVPA', 'Diabetes', 'Hypertension', 'Hyperlipidemia', 'Hepatatis']] = df_init[['delta_time', 'death', 'sex', 'Smoke', 'ALC', 'MVPA', 'Diabetes', 'Hypertension', 'Hyperlipidemia', 'Hepatatis']].astype(int)

model_expr = "delta_time ~ AGE + sex + BMI + Smoke + ALC + MVPA + Diabetes \
    + Hypertension + Hyperlipidemia + Hepatatis + max_heart_rate \
    + MBP + ABRP_CRF_tertile + death + delta_time"
y, X = dmatrices(model_expr, df_init, return_type='dataframe')

cph = CoxPHFitter().fit(X, 'delta_time', 'death')
print(cph.print_summary())
#cph.plot_partial_effects_on_outcome('ABRP_CRF', values=range(0, 14, 1) , cmap='coolwarm')
cph.plot_partial_effects_on_outcome(covariates=['ABRP_CRF_tertile[T.T2]', 'ABRP_CRF_tertile[T.T3]'], values=[1, 0], cmap='coolwarm')



#%%

df_tmp = pd.concat((df_init, pd.get_dummies(df_init['ABRP_CRF_tertile'], prefix='ABRP_CRF_tertile_tmp', drop_first=True)), axis=1)
# cox_var = ['AGE', 'sex', 'death', 'delta_time', 'Smoke', 'ALC', 'BMI', 'MVPA', 
#            'Diabetes', 'Hypertension', 'Hyperlipidemia', 'Hepatatis', 'max_heart_rate', 
#            'HDL-C', 'MBP', 'ABRP_CRF_tertile_nm']
cph = CoxPHFitter().fit(df_tmp, 'delta_time', 'death', strata=['ABRP_CRF_tertile_tmp'], formu)
#cph.plot_partial_effects_on_outcome('ABRP_CRF', values=range(0, 14, 1) , cmap='coolwarm')
cph.plot_partial_effects_on_outcome(covariates='ABRP_CRF_tertile_tmp', values=range(0, 14, 1) , cmap='coolwarm')





#%%
from lifelines import datasets, CoxPHFitter
cox_var = ['AGE', 'sex', 'death', 'delta_time', 'Smoke', 'ALC', 'BMI', 'MVPA', 
           'Diabetes', 'Hypertension', 'Hyperlipidemia', 'Hepatatis', 'max_heart_rate', 
           'HDL-C', 'MBP', 'ABRP_CRF_tertile_nm']
cph = CoxPHFitter().fit(df_init[cox_var], 'delta_time', 'death', strata=['ABRP_CRF_tertile_nm'])
#cph.plot_partial_effects_on_outcome('ABRP_CRF', values=range(0, 14, 1) , cmap='coolwarm')
cph.plot_partial_effects_on_outcome(covariates='ABRP_CRF_tertile_nm', values=range(0, 14, 1) , cmap='coolwarm')

# %% 
df_init.head()
df_init['ABRP_CRF_tertile_nm'].value_counts()

# %%
df = km.example_data()
time_event=df['time']
censoring=df['Died']
labx=df['group']

# Compute survival
out=km.fit(time_event, censoring, labx)
km.plot(out, cmap='Set2', methodtype='custom')
# %%
df
# %%
df_init['death'].value_counts()
# %%
df
# %%
df.info()
# %%
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter

rossi = load_rossi()
cph = CoxPHFitter().fit(rossi, 'week', 'arrest')

axes = cph.check_assumptions(rossi, show_plots=True)
# %%
from lifelines import datasets, CoxPHFitter
rossi = datasets.load_rossi()

cph = CoxPHFitter().fit(rossi, 'week', 'arrest')
cph.plot_partial_effects_on_outcome('prio', values=range(0, 15, 3), cmap='coolwarm')
# %%
