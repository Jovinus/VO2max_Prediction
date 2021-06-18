# %% Import Packages
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from IPython.display import display
# %% Read Data and Select Subset to Analysis
df_orig = pd.read_csv("../Results/MF_general_eq_for_surv.csv")
df_select = df_orig[df_orig['sex'] == False].reset_index(drop=True)
# %%
kmf = KaplanMeierFitter()

fig, ax = plt.subplots(1,1, figsize=(8, 8))

kmf.fit(durations=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T1', 'delta_time'], event_observed=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T1', 'death'], label='Q1')
kmf.plot_survival_function(ax=ax)

kmf.fit(durations=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T2', 'delta_time'], event_observed=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T2', 'death'], label='Q2')
kmf.plot_survival_function(ax=ax)

kmf.fit(durations=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T3', 'delta_time'], event_observed=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T3', 'death'], label='Q3')
kmf.plot_survival_function(ax=ax)

plt.title('Kaplan Meier Curve for estimated CRF Tertile')

# %%
kmf = KaplanMeierFitter()

fig, ax = plt.subplots(1,1, figsize=(8, 8))

kmf.fit(durations=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T1', 'delta_time'], event_observed=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T1', 'death'], label='Q1')
kmf.survival_function_.plot(ax=ax)

kmf.fit(durations=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T2', 'delta_time'], event_observed=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T2', 'death'], label='Q2')
kmf.survival_function_.plot(ax=ax)

kmf.fit(durations=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T3', 'delta_time'], event_observed=df_select.loc[df_select['ABRP_CRF_tertile'] == 'T3', 'death'], label='Q3')
kmf.survival_function_.plot(ax=ax)

plt.title('Kaplan Meier Curve for estimated CRF Tertile')
# %%
from lifelines.statistics import multivariate_logrank_test

results = multivariate_logrank_test(event_durations=df_select['delta_time'],
                                    groups=df_select['ABRP_CRF_tertile'],
                                    event_observed=df_select['death'])
print(results.print_summary)

# %%
from lifelines.statistics import pairwise_logrank_test
results = pairwise_logrank_test(event_durations=df_select['delta_time'],
                                groups=df_select['ABRP_CRF_tertile'],
                                event_observed=df_select['death'])
print(results.print_summary)
# %%
