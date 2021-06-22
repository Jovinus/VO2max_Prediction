# %%
import pandas as pd
from IPython.display import display
import numpy as np
import re
pd.set_option("display.max_columns", None)
# %%
df_cac = pd.read_csv('../Data/request/vo2max_cac.csv')
df_carotid = pd.read_csv('../Data/request/vo2max_carotid.csv')
df_pwv = pd.read_csv('../Data/request/vo2max_pwv_abi.csv')

# %%
df_pwv['SM_DATE'] = df_pwv['처방일자#4'].astype('datetime64')
df_carotid['SM_DATE'] = df_carotid['건진일자#2'].astype('datetime64')
df_cac['SM_DATE'] = df_cac['SM_DATE#2'].astype('datetime64')

#%%
df_cac = df_cac[['환자번호#1', 'SM_DATE', 'AJ_130_SCORE#3', 'VOLUME_SCORE#4']]
df_cac.rename(columns={'VOLUME_SCORE#4': 'Voluem_Score', 'AJ_130_SCORE#3': 'AJ_130_Score'}, inplace=True)

# %%
df_carotid['mean_IMT'] = df_carotid.loc[:, 'Carotid US CCA : IMT(Rt)#5':'Carotid US CCA : IMT(Lt)#6'].mean(axis=1)
mask_carotid = (df_carotid.loc[:, 'Carotid US CCA : IMT(Rt)#5':'Carotid US CCA : IMT(Lt)#6'].notnull().sum(axis=1) >= 2)
df_carotid = df_carotid[mask_carotid][['환자번호#1', 'SM_DATE', 'mean_IMT']].reset_index(drop=True)
# %%
df_pwv = pd.pivot_table(values='검사결과수치값#7', columns='검사명#6', index=['환자번호#1', 'SM_DATE'], data=df_pwv, aggfunc=lambda x: x).reset_index(drop=False)
df_pwv.columns.name = None

# %%
def split_data(input_x, option=0):
    
    if option != 0:
        option = 1
        
    splitted = input_x.split('/')
    
    try:
        return float(splitted[option])
    
    except (ValueError, IndexError):
        return np.nan

# %%
df_pwv['baPWV_Rt'] = df_pwv['baPWV(Rt/Lt)'].apply(lambda x: split_data(input_x=x, option=0))
df_pwv['baPWV_Lt'] = df_pwv['baPWV(Rt/Lt)'].apply(lambda x: split_data(input_x=x, option=1))
df_pwv['ABI_Rt'] = df_pwv['ABI(Rt/Lt)'].apply(lambda x: split_data(input_x=x, option=0))
df_pwv['ABI_Lt'] = df_pwv['ABI(Rt/Lt)'].apply(lambda x: split_data(input_x=x, option=1))

df_pwv['mean_baPWV'] = df_pwv[['baPWV_Lt', 'baPWV_Rt']].mean(axis=1)
df_pwv['mean_ABI'] = df_pwv[['ABI_Lt', 'ABI_Rt']].mean(axis=1)

df_pwv = df_pwv[['환자번호#1', 'SM_DATE' ,'mean_baPWV', 'mean_ABI']]
# %%
display(df_pwv)
display(df_cac)
display(df_carotid)
# %%
df_results = pd.merge(df_pwv, df_carotid, on=['환자번호#1', 'SM_DATE'], how='outer')
df_results = pd.merge(df_results, df_cac, on=['환자번호#1', 'SM_DATE'], how='outer')
df_results.rename(columns={'환자번호#1':'ID'}, inplace=True)
# %%
display(df_results)
# %%

df_mets = pd.read_csv("../Data/general_eq.csv")
df_id_list = pd.read_excel("../Data/raw_data/VO2peak_HPCID.xlsx")
df_mets = pd.merge(df_mets, df_id_list, on='HPCID', how='left')
df_mets['SM_DATE'] = df_mets['SM_DATE'].astype('datetime64')
# %%
df_final = pd.merge(df_mets, df_results, 
                    left_on=['CDW_NO', 'SM_DATE'], 
                    right_on=['ID', 'SM_DATE'], how='left')
# %%
df_final.to_csv("./data_request.csv", index=False, encoding='utf-8-sig')
# %%
