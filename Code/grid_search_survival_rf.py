# %% import package to use
import datatable
import pandas as pd
import xgboost as xgb
from IPython.display import display
import numpy as np
import os
from pysurvival.models.survival_forest import RandomSurvivalForestModel
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

hyper_param_max_features = ['sqrt', 'log2', 'all']
hyper_param_num_trees = [10, 50, 100 ,200]


results = {}

for hyper_max_features in tqdm(hyper_param_max_features, desc= 'max_features'):

    for hyper_num_trees in tqdm(hyper_param_num_trees, desc= 'num_trees'):

        skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=20)

        scores = []

        for train_index, validation_index in skf.split(train_set, train_set['death']):
            
                X_train = train_set.iloc[train_index][feature_names]
                X_validation = train_set.iloc[validation_index][feature_names]

                model_rf = RandomSurvivalForestModel(num_trees=hyper_num_trees)

                model_rf.fit(X=train_set.iloc[train_index][feature_names], 
                             T=train_set.iloc[train_index]['delta_time'], 
                             E=train_set.iloc[train_index]['death'], 
                             max_features=hyper_max_features)
                
                c_index = concordance_index(model=model_rf, 
                            X=train_set.iloc[validation_index][feature_names], 
                            T=train_set.iloc[validation_index]['delta_time'],
                            E=train_set.iloc[validation_index]['death'])

                scores.append(c_index)

        result = {'max_features': hyper_max_features, 'num_trees': hyper_num_trees, 'scores':scores, 'mean_score':np.mean(scores), 'std_score':np.std(scores)}
        # print(result['mean_score'])
        results["max_features_" + str(hyper_max_features) + "_num_trees_" + str(hyper_num_trees)] = result

# %%
with open('../Model/MF_20_survival_rf_results.pickle', 'wb') as file_nm:
    pickle.dump(results, file_nm, protocol=pickle.HIGHEST_PROTOCOL)
# %%
