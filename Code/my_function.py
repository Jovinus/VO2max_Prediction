import pandas as pd
import numpy as np
from IPython.display import display
import datatable
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
pd.set_option("display.max.columns", None)


work_path = '/Users/lkh256/PycharmProjects/VO2max_Prediction/'
data_list = glob.glob(os.path.join(work_path, 'Data/raw_data/DATA*'))
print(data_list)



df_data = pd.DataFrame()
for nm_data in data_list:
    df_data = df_data.append(datatable.fread(nm_data, encoding='CP949', na_strings=['NA', '']).to_pandas())
print(len(df_data))

## HPCID -> CDW_NO
df_idcon = pd.read_excel(os.path.join(work_path, "Data/raw_data/VO2peak_HPCID.xlsx"))
df_idcon.head()

df_data = pd.merge(df_data, df_idcon)
print(len(df_data))

df_data.columns = df_data.columns.str.lower()
df_data['sm_date'] = df_data['sm_date'].astype('datetime64')

len(set(df_data['hpcid']))