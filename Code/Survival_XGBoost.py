# %% import package to use
import datatable
import pandas as pd
import xgboost as xgb
from IPython.display import display
import numpy as np
from xgbse.metrics import concordance_index
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
                 'Hyperlipidemia', 'HDL_C', 'LDL_C', 'MBP', 'ABRP_CRF']

"""
I have to add
age-only, sex-only, no-age_sex  
"""
# %%
from sklearn.model_selection import train_test_split
train_set, validation_set = train_test_split(df_orig, 
                                             train_size=0.7, 
                                             stratify=df_orig['death'], 
                                             random_state=10)

X = df_orig[feature_names].values
dallset = xgb.DMatrix(X)
dallset.set_float_info('label_lower_bound', df_orig['lower_bound'].values)
dallset.set_float_info('label_upper_bound', df_orig['upper_bound'].values)

X_train = train_set[feature_names].values
dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info('label_lower_bound', train_set['lower_bound'].values)
dtrain.set_float_info('label_upper_bound', train_set['upper_bound'].values)

X_validation = validation_set[feature_names].values
dvalidation = xgb.DMatrix(X_validation)
dvalidation.set_float_info('label_lower_bound', validation_set['lower_bound'].values)
dvalidation.set_float_info('label_upper_bound', validation_set['upper_bound'].values)

# %%
params = {'objective': 'survival:aft',
          'eval_metric': 'aft-nloglik',
          'aft_loss_distribution': 'normal',
          'aft_loss_distribution_scale': 1.20,
          'tree_method': 'gpu_hist', 
          'gpu_id': '0',
          'learning_rate': 0.001, 
          'max_depth': 4}
model_xgb = xgb.train(params, dtrain, 
                      num_boost_round=15000, 
                      evals=[(dvalidation, 'validation')], 
                      verbose_eval=100, 
                      early_stopping_rounds=1000)
# %%
model_xgb.set_param({"predictor": "gpu_predictor"})
shap_values = model_xgb.predict(dtrain, pred_contribs=True)
# %%
shap_interaction_values = model_xgb.predict(dtrain, pred_interactions=True)
# %%
import shap

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_validation)

# visualize the first prediction's explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_validation[0, :],
    feature_names=feature_names,
    matplotlib=True
)
# %%
shap.summary_plot(shap_values, X_validation, plot_type="bar", feature_names=feature_names)
# %%
shap.summary_plot(shap_values, X_validation, feature_names=feature_names)

# %%
from lifelines.utils import concordance_index
concordance_index(validation_set['lower_bound'].values, 
                  model_xgb.predict(dvalidation), 
                  event_observed=validation_set['death'].astype(int))
# %%

# %%

# %%
