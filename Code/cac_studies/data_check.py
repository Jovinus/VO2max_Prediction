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
DATA_PATH = '/home/lkh256/Studio/VO2max_Prediction/Data/raw_data'
df_init = pd.DataFrame([])
for i in glob.glob(os.path.join(DATA_PATH, 'DATA*.csv')):
    df_init = pd.concat((df_init, datatable.fread(i, encoding='CP949', na_strings=['', 'NA']).to_pandas()), axis=0)
    print(df_init.shape)
df_init = df_init.sort_values(['HPCID', 'SM_DATE']).reset_index(drop=True)
df_init['SM_DATE'] = df_init['SM_DATE'].astype('datetime64')

print("Number of Case = {}".format(len(df_init)))
print("Number of patients = {}".format(len(set(df_init['HPCID']))))
display(df_init.head(1))

# %%
df_death = pd.read_excel(os.path.join(DATA_PATH, 'HPC_death_2019_06_18.xlsx'))
df_death['death_date'] = df_death['사망일'].astype('datetime64')
df_id = pd.read_excel(os.path.join(DATA_PATH, 'VO2peak_HPCID.xlsx'))
display(df_death.head(), df_id.head())
# %%
df_init = pd.merge(df_init, df_id, left_on=['HPCID'], right_on=['HPCID'], how='left')
df_init = pd.merge(df_init, df_death[['ptno', 'death_date']], left_on=['CDW_NO'], right_on=['ptno'], how='left').drop(columns='ptno')
display(df_init.head())
# %% Define derived variables 

#### Survival variable
df_init['death'] = np.where(df_init['death_date'].notnull(), 1, 0)
df_init['delta_time'] = np.where(df_init['death'] == 1, (df_init['death_date'] - df_init['SM_DATE']).dt.days, (np.datetime64('2019-06-18') - df_init['SM_DATE']).dt.days)

#### Code 1 to row that have GXT
df_init['gxt_ys'] = np.where(df_init['SM3720'].notnull(), 1, 0)

#### Calculate BMI
def cal_bmi_crf_asmi(df):
    df["BMI_cal"] = df.apply(lambda x: x['SM0102']/((x['SM0101']/100)**2), axis=1)
    df['VO2max'] = df['SM3720'] * 3.5
    df['ASMI'] = df[['SM0151', 'SM0152', 'SM0154', 'SM0155']].sum(axis=1) / ((df['SM0101']/100)**2)
    df['MBP'] = df['SM0600DBP'] + 0.4 * (df['SM0600SBP'] - df['SM0600DBP'])
    return df

#### Smoking
df_init['Smoke'] = np.where(df_init['SMK'] == 2, 1, 0)

#### Alcohol
df_init['ALC'] = np.where(df_init['ALC_YS'] == 1, 1, 0)

#### Sex
df_init['sex'] = np.where(df_init['GEND_CD'] == 'M', 0, 1)

#### Select rows that RER over the goldenstandard
def is_golden_RER(df):
    ## RER >= 1.1 -> Golden Standard를 만족하는 경우
    df.loc[:, "max_RER"] = df.loc[:, "SM3691":"SM3697"].max(axis=1)
    df.loc[:, "max_heart_rate"] = df.loc[:, "SM3631":"SM3637"].max(axis=1)
    df['RER_over_gs'] = np.where(df['max_RER'] >= 1, 1, 0)
    print("RER >= 1.1: n = ", len(set(df[df["RER_over_gs"] == 1]["HPCID"])))
    #df = df.loc[df["RER_over_gs"] == 1]
    return df.drop(columns=['max_RER'])

#### Extract feature from self-reported physical exercise queation -> MVPA(yes/no)
def is_MVPA(df):
    ## epidemiology version
    df['OVERALL_PHYSICAL_ACTIVITY'] = np.where(df['OVERALL_PHYSICAL_ACTIVITY'].isnull()|df['OVERALL_PHYSICAL_ACTIVITY'].isin([9999]), 0, df['OVERALL_PHYSICAL_ACTIVITY'])
    df['PHY_DURATION'] = np.where(df['PHY_DURATION'].isnull()|df['PHY_DURATION'].isin([9999]), 0 ,df['PHY_DURATION'])
    df['PHY_FREQ'] = np.where(df['PHY_FREQ'].isnull()|df['PHY_FREQ'].isin([9999]), 0, df['PHY_FREQ'])

    dur_mapper = {0:0, 1:10, 2:30, 3:50, 4:60}
    df['tmp_phy_duration'] = df['PHY_DURATION'].map(dur_mapper)

    freq_mapper = {0:0, 1:1.5, 2:3.5, 3:6}
    df['tmp_phy_freq'] = df['PHY_FREQ'].map(freq_mapper)

    df['tmp_phy_act'] = df['tmp_phy_freq'] * df['tmp_phy_duration']
    
    df['tmp_act_mets'] = df['OVERALL_PHYSICAL_ACTIVITY'].map({0:0, 1:3.3, 1:4, 2:8})
    df['METs_week'] =  df['tmp_act_mets'] * df['tmp_phy_act']
    df['METs_week'] = np.where(df['METs_week'].isnull(), 0, df['METs_week'])
    

    #### Define MVPA(yes = 1/no = 0)
    df['MVPA'] = np.where(df['OVERALL_PHYSICAL_ACTIVITY'].isin([1, 2]) & (df['tmp_phy_act'] >= 150), 1, 0)

    display(df['MVPA'].value_counts(dropna=False))
    
    return df.drop(columns=['tmp_phy_duration', 'tmp_phy_freq', 'tmp_phy_act', 'tmp_act_mets'])

#### Define disease to exclude
def have_disease(df):

    print("\nInitial n: ", len(set(df["HPCID"])))

    #### Diabetes
    df['Diabetes'] = np.where((df['BL3118'] >= 126) | (df['BL3164'] >= 6.5) | (df['MED_DIABETES'] == 1) | (df['TRT_DIABETES'] == 0) | (df['TRT_MED_DIABETES'] == 1), 1, 0)
    print("\nDiabetes: n = ", len(set(df[df["Diabetes"] == 1]["HPCID"])))
    
    #### Hypertension
    df['Hypertension'] = np.where((df['SM0600SBP'] >= 140) | (df['SM0600DBP'] >= 90) | (df['HISTORY_HYPERTENSION'] == 1) | (df['TRT_HYPERTENSION'].isin([0, 1])) , 1, 0)
    df['HTN_med'] = np.where((df['Hypertension'] == 1) & (df['MED_HYPERTENSION'] == 1), 1, 0)
    print("\nHypertension: n = ", len(set(df[df["Hypertension"] == 1]["HPCID"])))
    
    #### Hyperlipidemia
    df['Hyperlipidemia'] = np.where((df['BL314201'] > 130) | (df['BL3142'] < 40) | (df['MED_HYPERLIPIDEMIA'] == 1) | (df['TRT_MED_HYPERLIPIDEMIA'].isin([0, 1])), 1, 0)
    print("\nHyperlipidemia: n = ", len(set(df[df["Hyperlipidemia"] == 1]["HPCID"])))
    
    #### Hepatatis
    df['Hepatatis'] = np.where((df['BL5111'] > 0.5) | (df['BL5115'] >= 0.5), 1, 0)
    print("\nHepatatis: n = ", len(set(df[df["Hepatatis"] == 1]["HPCID"])))

    #####################################Roughly Defined Chronic Disease#################################
    
    #### Stroke
    df['Stroke'] = df[['HISTORY_STROKE', 'TRT_STROKE', 'STATUS_STROKE', 'TRT_STROKE_OP']].any(axis=1, skipna=True) * 1
    display(df['Stroke'].value_counts())
    print("\nStroke: n = ",len(set(df[df['Stroke'] == 1]["HPCID"])))
    
    #### Angina
    df['Angina'] = df[['HISTORY_ANGINA', 'TRT_ANGINA', 'STATUS_ANGINA', 'TRT_ANGINA_OP']].any(axis=1, skipna=True) * 1
    display(df['Angina'].value_counts())
    print("\nAngina: n = ",len(set(df[df["Angina"] == 1]["HPCID"])))
    
    #### MI
    df['MI'] = df[['HISTORY_MI', 'TRT_MI', 'STATUS_MI', 'TRT_MI_OP']].any(axis=1, skipna=True) * 1
    display(df["MI"].value_counts())
    print("\nMyocardiac Infraction: n = ", len(set(df[df['MI'] == 1]["HPCID"])))
    
    #### Asthma
    df['Asthma'] = df[['HISTORY_ASTHMA']].any(axis=1, skipna=True) * 1
    print("\nAsthma: n = ",len(set(df[df["Asthma"] == 1]["HPCID"])))
    
    #### Cancer
    df['Cancer'] = df.loc[:, "HISTORY_CANCER": "TRT_CANCER_OTHER_OT"].any(axis=1, skipna=True)*1
    print("\nExclude_cancer n: ", len(set(df[df["Cancer"] == 1]["HPCID"])))
    
    return df

# %% Processing the dataset

df_init = cal_bmi_crf_asmi(df_init)
df_init = is_golden_RER(df_init)
df_init = is_MVPA(df_init) 
df_init = have_disease(df_init)

# %% 
df_init.head()
# %%
plt.figure(figsize=(10, 10))
sns.histplot(data=df_init[(df_init['RER_over_gs'] == 1) & (df_init['METs_week'] > 0)], x='METs_week', kde=True)
plt.show()
# %%
#### METs distribution
df_init[df_init['RER_over_gs'] == 1]['METs_week'].value_counts().sort_index()
# %%
df_init[df_init['RER_over_gs'] == 1]['HPCID'].value_counts().value_counts()
# %%
