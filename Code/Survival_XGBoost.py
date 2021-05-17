# %% import package to use
import datatable
import pandas as pd
import xgboost as xgb
from IPython.display import display
import numpy as np
pd.set_option('display.max_columns', None)
# %%
DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Results"
df_orig = datatable.fread(os.path.join(DATA_PATH, 'MF_general_eq_for_surv.csv'), 
                          na_strings=['', 'NA']).to_pandas()

# %% Data preprocessing for Survival XGBoost

df_orig['lower_bound'] = df_orig['delta_time']
df_orig['upper_bound'] = np.where(df_orig['death'] == 1, df_orig['lower_bound'], +np.inf)
df_orig.head()
# %%
feature_names = ['AGE', 'sex', 'Smoke', 'ALC', 'CRP', 'TG', 'Diabetes', 'Hypertension', 
                 'Hyperlipidemia', 'HDL_C', 'LDL_C', 'MBP', 'APRP_CRF']

"""
I have to add
age-only, sex-only, no-age_sex  
"""
# %%
xgb.set_config(verbosity=0)

X = df_orig[feature_names].values
dtrain = xgb.DMatrix(X)

y_lower_bound = df_orig['lower_bound'].values
y_upper_bound = df_orig['upper_bound'].values

dtrain.set_float_info('label_lower_bound', y_lower_bound)
dtrain.set_float_info('label_upper_bound', y_upper_bound)
# %%
params = {'objective': 'survival:aft',
          'eval_metric': 'aft-nloglik',
          'aft_loss_distribution': 'normal',
          'aft_loss_distribution_scale': 1.20,
          'tree_method': 'gpu_hist', 
          'gpu_id': '0',
          'learning_rate': 0.001, 
          'max_depth': 5}
bst = xgb.train(params, dtrain, num_boost_round=15000,
                evals=[(dtrain, 'train')])
# %%
bst.set_param({"predictor": "gpu_predictor"})
shap_values = bst.predict(dtrain, pred_contribs=True)
# %%
shap_interaction_values = bst.predict(dtrain, pred_interactions=True)
# %%
import shap

# shap will call the GPU accelerated version as long as the predictor parameter is set to "gpu_predictor"
bst.set_param({"predictor": "gpu_predictor"})
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X[0, :],
    feature_names=feature_names,
    matplotlib=True
)
# %%
shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names)
# %%
shap.summary_plot(shap_values, X, feature_names=feature_names)
# %%

# %%
