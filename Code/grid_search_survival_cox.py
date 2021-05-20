# %% import package to use
import datatable
import pandas as pd
from IPython.display import display
import numpy as np
import os
from pysurvival.models.semi_parametric import CoxPHModel
pd.set_option('display.max_columns', None)
# %%
DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Results"
df_orig = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_for_surv.csv'), na_strings=['', 'NA']).to_pandas()
df_orig['lower_bound'] = df_orig['delta_time']
df_orig['upper_bound'] = np.where(df_orig['death'] == 1, df_orig['lower_bound'], +np.inf)
df_orig.head()

# %%
feature_names = ['AGE', 'sex', 'Smoke', 'ALC', 'CRP', 'TG', 'Diabetes', 'Hypertension', 
                 'Hyperlipidemia', 'HDL_C', 'LDL_C', 'MBP', 'ABRP_CRF']

"""
Option 1: Adjust age
Option 2: Adjust sex
Option 3: Adjust age and sex
Option 3: Non-adjust age and sex
"""

# %% 
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_orig, train_size=0.8, stratify=df_orig['death'], random_state=1005)

from tqdm import tqdm
from pysurvival.utils.metrics import concordance_index
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import pickle

hyper_param_l2 = [0.5, 0.1]
hyper_param_lr = [0.1]


results = {}

for hyper_l2 in tqdm(hyper_param_l2, desc= 'l2'):

    for hyper_lr in tqdm(hyper_param_lr, desc= 'l_rate'):

        skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

        scores = []

        for train_index, validation_index in skf.split(train_set, train_set['death']):
            
                X_train = train_set.iloc[train_index][feature_names]
                X_validation = train_set.iloc[validation_index][feature_names]

                model_coxph = CoxPHModel()

                model_coxph.fit(X=train_set.iloc[train_index][feature_names], 
                                T=train_set.iloc[train_index]['delta_time'], 
                                E=train_set.iloc[train_index]['death'], 
                                init_method='orthogonal',
                                lr=hyper_lr, 
                                max_iter=1000, 
                                l2_reg=hyper_l2,
                                verbose=False)
                
                c_index = concordance_index(model=model_coxph, 
                            X=train_set.iloc[validation_index][feature_names], 
                            T=train_set.iloc[validation_index]['delta_time'],
                            E=train_set.iloc[validation_index]['death'])

                scores.append(c_index)

        result = {'l2_reg': hyper_l2, 'learning_rate': hyper_lr, 'scores':scores, 'mean_score':np.mean(scores), 'std_score':np.std(scores)}
        # print(result['mean_score'])
        results["l2_reg_" + str(hyper_l2) + "_lr_" + str(hyper_lr)] = result

# %%
with open('../Model/MF_20_survival_coxph_results.pickle', 'wb') as file_nm:
    pickle.dump(results, file_nm, protocol=pickle.HIGHEST_PROTOCOL)
# %%
