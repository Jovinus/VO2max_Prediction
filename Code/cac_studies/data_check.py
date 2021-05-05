# %% Import package to use
import pandas as pd
import datatable
import os
import glob
import numpy as np
from IPython.display import display
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

# %% Read Data for preprocessing
DATA_PATH = '/home/lkh256/Studio/VO2max_Prediction/Data/'
df_init = datatable.fread(os.path.join(DATA_PATH, 'processed_whole_set.csv'), encoding='utf-8-sig', na_strings=['', 'NA']).to_pandas()

df_init.head()
# %%
#### METs per week distribution

plt.figure(figsize=(10, 10))
sns.histplot(data=df_init[(df_init['RER_over_gs'] == 1) & (df_init['METs_week'] > 0)], x='METs_week', kde=True)
plt.show()

print(df_init[df_init['RER_over_gs'] == 1]['METs_week'].value_counts().sort_index())

# %%
#### RER cut off 1.0 was used 
df_init[df_init['RER_over_gs'] == 1]['HPCID'].value_counts().value_counts()

# %%
#### CAC Score
print(len(df_init[df_init['RER_over_gs'] == 1]))
print(df_init[df_init['RER_over_gs'] == 1]['RC118401'].notnull().sum())
print(df_init[df_init['RER_over_gs'] == 1]['RC118402'].notnull().sum())

#### 검사 건수
print(df_init[(df_init['RER_over_gs'] == 1) & (df_init['RC118401'].notnull())]['HPCID'].value_counts().value_counts())
print(df_init[(df_init['RER_over_gs'] == 1) & (df_init['RC118402'].notnull())]['HPCID'].value_counts().value_counts())

# %%
#### METs Only

df_select = df_init[df_init['RER_over_gs'] == 1].reset_index(drop=True)
#### AJT-130 score
print(df_select[df_select['RC118401'].notnull()]['HPCID'].value_counts().value_counts())
#### Volume score
print(df_select[df_select['RC118402'].notnull()]['HPCID'].value_counts().value_counts())

# %%

#### Questionnaire
df_select = df_init
#### AJT-130 score
print(df_select[df_select['RC118401'].notnull()]['HPCID'].value_counts().value_counts())
#### Volume score
print(df_select[df_select['RC118402'].notnull()]['HPCID'].value_counts().value_counts())
# %%

print(df_init['METs_week'].value_counts())

df_select = df_init[df_init['METs_week'] > 0]
tmp = (df_init['METs_week'] > 0).map({True:"isn't_zoro", False:'is_zero'})

plt.figure(figsize=(10, 10))
sns.countplot(x=tmp)
plt.show()

plt.figure(figsize=(10, 10))
sns.histplot(data=df_select, x='METs_week', kde=True)
plt.show()

#### AJT-130 score
print(df_select[df_select['RC118401'].notnull()]['HPCID'].value_counts().value_counts())
#### Volume score
print(df_select[df_select['RC118402'].notnull()]['HPCID'].value_counts().value_counts())

#### 검사 건수
print("검사 빈도")
print(df_select['HPCID'].value_counts().value_counts())

#### 검사 건수
print("환산 값이 있으면서 CAC가 있는 경우")
print(df_select[(df_select['METs_week'] > 0) & (df_select['RC118401'].notnull())]['HPCID'].value_counts().value_counts())
print(df_select[(df_select['METs_week'] > 0) & (df_select['RC118402'].notnull())]['HPCID'].value_counts().value_counts())
