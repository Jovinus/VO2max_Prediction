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
DATA_PATH = '/home/lkh256/Studio/VO2max_Prediction/Data/Survival_set'
df_init = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_survival.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()
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
    + MBP + ABR_CRF_tertile + death + delta_time"
y, X = dmatrices(model_expr, df_init, return_type='dataframe')

cph = CoxPHFitter().fit(X, 'delta_time', 'death')
print(cph.print_summary(style='ascii'))
#cph.plot_partial_effects_on_outcome('ABRP_CRF', values=range(0, 14, 1) , cmap='coolwarm')
plt.figure(figsize=(20, 20))
ax = cph.plot_partial_effects_on_outcome(covariates=['ABR_CRF_tertile[T.T2]', 'ABR_CRF_tertile[T.T3]'], 
                                    values=[(1, 0), (0, 1)])
ax.plot(line_width=2)

# %%

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
cph = CoxPHFitter().fit(X, 'delta_time', 'death')
cph.plot_partial_effects_on_outcome(covariates=['ABR_CRF_tertile[T.T2]', 'ABR_CRF_tertile[T.T3]'], 
                                    values=[(1, 0), (0, 1)], ax=ax)
plt.legend(['T2', 'T3', 'T1'])
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
cph = CoxPHFitter().fit(X, 'delta_time', 'death')
cph.plot(ax=ax, hazard_ratios=True)
plt.show()
# %%
