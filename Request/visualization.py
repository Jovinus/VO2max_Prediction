# %%
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
pd.set_option("display.max_columns", None)
# %%
df_orig = pd.read_csv("./data_request.csv")
df_orig.head()

# %%
df_orig['sex'].value_counts()
df_orig['Pulse_Pressure'] = df_orig['SBP'] - df_orig['DBP']
df_orig = df_orig[(df_orig['Hypertension'] != 1) & (df_orig['Diabetes'] != 1)]
# df_orig = df_orig[(df_orig['Hypertension'] != 1)]
# %%
sns.lineplot(x='AGE', y='mean_baPWV', data=df_orig)
# %% male
df_male = df_orig[df_orig['sex'] == 0].reset_index(drop=True)
bins = [10, 30, 40, 50, 60, 70, 80]
df_male['Age'] = pd.cut(df_male['AGE'], bins=bins)

mapper = {"(10, 30]":25, "(30, 40]":35, "(40, 50]":45, "(50, 60]":55, '(60, 70]':65, '(70, 80]':75}
df_male['Age'] = df_male['Age'].astype(str).map(mapper).astype(float)
df_male.rename(columns={'mean_IMT':'CAROTID IMT', 'mean_baPWV':"baPWV", 
                        'AJ_130_Score':"AJ-130 Score", "Volume_Score":"Volume Score", 
                        "Pulse_Pressure":"Pulse Pressure", 'mean_ABI':'ABI'}, inplace=True)

# %% female
df_female = df_orig[df_orig['sex'] == 1].reset_index(drop=True)
bins = [20, 30, 40, 50, 60, 70, 80]
df_female['Age'] = pd.cut(df_female['AGE'], bins=bins)

mapper = {"(20, 30]":25, "(30, 40]":35, "(40, 50]":45, "(50, 60]":55, '(60, 70]':65, '(70, 80]':75}
df_female['Age'] = df_female['Age'].astype(str).map(mapper).astype(float)
df_female.rename(columns={'mean_IMT':'CAROTID IMT', 'mean_baPWV':"baPWV", 
                        'AJ_130_Score':"AJ-130 Score", "Volume_Score":"Volume Score", 
                        "Pulse_Pressure":"Pulse Pressure", 'mean_ABI':'ABI'}, inplace=True)

# %%
fig, ax = plt.subplots(1, figsize=(10,10))
sns.regplot(x='AGE', y='baPWV', data=df_male, ax=ax, scatter=False, order=3, ci=95, label='Male')
sns.regplot(x='AGE', y='baPWV', data=df_female, ax=ax, scatter=False, order=3, ci=95, label='Female')
plt.legend()
plt.show() 
# %%
print(len(df_male[(df_male['Age'] < 40) & df_male['baPWV'].notnull()]))
# %%
print(len(df_female[(df_female['Age'] < 40) & df_female['baPWV'].notnull()]))
# %%
df_male[df_male['AGE'] < 30]['baPWV'].mean()
df_male[df_male['AGE'] < 30]['baPWV'].std()
# %%
