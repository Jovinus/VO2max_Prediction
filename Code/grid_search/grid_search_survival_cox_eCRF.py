# %% import package to use
import datatable
import pandas as pd
import xgboost as xgb
from IPython.display import display
import numpy as np
import os
from sksurv.linear_model import CoxPHSurvivalAnalysis
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import pickle
pd.set_option('display.max_columns', None)
# %%
DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Results"
df_orig = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_for_surv.csv'), na_strings=['', 'NA']).to_pandas()
df_orig['lower_bound'] = df_orig['delta_time']
df_orig['upper_bound'] = np.where(df_orig['death'] == 1, df_orig['lower_bound'], +np.inf)
df_orig.head()

# %%
feature_names = ['AGE', 'sex', 'Smoke', 'ALC', 'CRP', 'TG', 'Diabetes', 
                 'Hypertension', 'Hyperlipidemia', 'HDL_C', 'LDL_C', 'MBP', 'ABRP_CRF']

feature_names = ['AGE', 'sex', 'Smoke', 'ALC', 'CRP', 'TG', 'Diabetes', 'HDL_C', 'LDL_C', 'MBP', 'ABRP_CRF']

"""
Option 1: Adjust age
Option 2: Adjust sex
Option 3: Adjust age and sex
Option 3: Non-adjust age and sex
"""

# %% 
from sklearn.model_selection import train_test_split

X_data = df_orig[feature_names]
y_data = np.zeros(df_orig.shape[0], dtype={'names':('death', 'delta_time'), 
                                           'formats':(bool, float)})
y_data['death'] = df_orig['death'].astype(bool)
y_data['delta_time'] = df_orig['delta_time'].astype(float)


X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, train_size=0.8, stratify=y_data['death'], random_state=1005)

hyper_param_ties= ['breslow', 'efron']
hyper_param_alpha = [0, 0.1, 0.2, 0.5, 1, 1.25, 1.5 ,2]


results = {}

for hyper_ties in tqdm(hyper_param_ties, desc= 'ties'):

    for hyper_alpha in tqdm(hyper_param_alpha, desc= 'alpha'):

        skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

        scores = []

        for train_index, validation_index in skf.split(X_train_data, y_train_data['death']):
            
                X_train = X_train_data.iloc[train_index][feature_names]
                X_validation = X_train_data.iloc[validation_index][feature_names]

                model_coxph = CoxPHSurvivalAnalysis(ties = hyper_ties, 
                                                    alpha = hyper_alpha)

                model_coxph.fit(X=X_train_data.iloc[train_index][feature_names], y=y_train_data[train_index])
                
                c_index = model_coxph.score(X_train_data.iloc[validation_index][feature_names], y_train_data[validation_index])

                scores.append(c_index)

        result = {'ties': hyper_ties, 'alpha': hyper_alpha, 'scores':scores, 'mean_score':np.mean(scores), 'std_score':np.std(scores)}
        # print(result['mean_score'])
        results["ties_" + str(hyper_ties) + "_alpha_" + str(hyper_alpha)] = result

# %%
with open('../../Model/MF_10_survival_coxph_results.pickle', 'wb') as file_nm:
    pickle.dump(results, file_nm, protocol=pickle.HIGHEST_PROTOCOL)
# %%
