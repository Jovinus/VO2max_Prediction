# %%
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
pd.set_option("display.max_columns", None)
# %%
df_orig = pd.read_csv("./data_request.csv")
df_orig.head()

# %%
df_orig['sex'].value_counts()
df_orig['Pulse_Pressure'] = df_orig['SBP'] - df_orig['DBP']
df_orig = df_orig[(df_orig['Hypertension'] != 1) & (df_orig['Diabetes'] != 1)]
df_orig = df_orig[(df_orig['Hypertension'] != 1)]
# %%
sns.lineplot(x='AGE', y='mean_baPWV', data=df_orig)
# %%
df_male = df_orig[df_orig['sex'] == 1].reset_index(drop=True)

# %%
df_male.info()

# %%
bins = [20, 30, 40, 50, 60, 70]
df_male['Age'] = pd.cut(df_male['AGE'], bins=bins)

mapper = {"(20, 30]":25, "(30, 40]":35, "(40, 50]":45, "(50, 60]":55, '(60, 70]':65}
df_male['Age'] = df_male['Age'].astype(str).map(mapper).astype(float)

# %%
df_male[['CRF_cat']] = pd.qcut(df_male['CRF'], q=3, labels=['Low', 'Moderate', 'High'], retbins=True)[0]
# %% lineplot
fig, ax = plt.subplots(1, figsize=(10,10))
sns.lineplot(x='AGE', y='mean_IMT', data=df_male, hue='CRF_cat', ax=ax)
plt.show()

fig, ax = plt.subplots(1, figsize=(10,10))
sns.lineplot(x='AGE', y='mean_baPWV', data=df_male, hue='CRF_cat', ax=ax)
plt.show()

# %% lineplot
fig, ax = plt.subplots(1, figsize=(10,10))
sns.lineplot(x='Age', y='mean_IMT', data=df_male, hue='CRF_cat', ax=ax)
plt.show()

fig, ax = plt.subplots(1, figsize=(10,10))
sns.lineplot(x='Age', y='mean_baPWV', data=df_male, hue='CRF_cat', ax=ax)
plt.show()



# %%
def plot_line(data, x_var, y_var, indi):
    
    subset = data[data[y_var].notnull()]
    
    subset['CRF_cat_age'] = subset.groupby('Age')['CRF'].apply(lambda x: pd.qcut(x=x, q=3, labels=['Low', 'Moderate', 'High']))

    
    fig, ax = plt.subplots(1, figsize=(20,10))
    sns.regplot(x=x_var, y=y_var, data=subset[subset[indi] == 'High'], ax=ax, scatter=False, label='High', order=5, ci=0)
    sns.regplot(x=x_var, y=y_var, data=subset[subset[indi] == 'Moderate'], ax=ax, scatter=False, label='Moderate', order=5, ci=0)
    sns.regplot(x=x_var, y=y_var, data=subset[subset[indi] == 'Low'], ax=ax, scatter=False, label='Low', order=5, ci=0)
    sns.regplot(x=x_var, y=y_var, data=subset, ax=ax, scatter=False, label='Average', order=5, ci=0)
    plt.legend()
    plt.show()


# %%
plot_line(data=df_male, x_var='Age', y_var='mean_IMT', indi='CRF_cat_age')
plot_line(data=df_male, x_var='Age', y_var='mean_baPWV', indi='CRF_cat_age')

# %%
from sklearn.linear_model import LinearRegression

def tertile_plot(dataset, outcome):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    linear_reg = LinearRegression()
    
    subset = dataset[dataset[outcome].notnull()]
    
    subset['CRF_cat_age'] = subset.groupby('Age')['CRF'].apply(lambda x: pd.qcut(x=x, q=3, labels=['Low', 'Moderate', 'High']))
    
    for labels in ['High', 'Moderate', 'Low']:
        X = subset[(subset[outcome].notnull()) & (subset['CRF_cat_age'] == labels)]['AGE'].values.reshape(-1, 1)
        y = subset[(subset[outcome].notnull()) & (subset['CRF_cat_age'] == labels)][outcome].values.reshape(-1, 1)

        linear_reg.fit(X, y)

        tmp_x = np.array(range(10, 80, 1)).reshape(-1, 1)

        tmp_y = linear_reg.predict(tmp_x)

        plt.plot(tmp_x, tmp_y, label=labels)
    
    plt.legend()
    plt.show()
# %%
df_male.rename(columns={'mean_IMT':'CAROTID IMT', 'mean_baPWV':"baPWV", 'AJ_130_Score':"AJ-130 Score", "Volume_Score":"Volume Score", "Pulse_Pressure":"Pulse Pressure", 'mean_ABI':'ABI'}, inplace=True)

# %%
plot_line(data=df_male, x_var='Age', y_var='CAROTID IMT', indi='CRF_cat_age')
plot_line(data=df_male, x_var='Age', y_var='baPWV', indi='CRF_cat_age')
plot_line(data=df_male, x_var='Age', y_var='ABI', indi='CRF_cat_age')
plot_line(data=df_male[df_male['AJ-130 Score'] > 0], x_var='Age', y_var='AJ-130 Score', indi='CRF_cat_age')
plot_line(data=df_male[df_male['Volume Score'] > 0], x_var='Age', y_var='Volume Score', indi='CRF_cat_age')
# %%
plot_line(data=df_male, x_var='Age', y_var='max_heart_rate', indi='CRF_cat_age')
plot_line(data=df_male, x_var='Age', y_var='Pulse Pressure', indi='CRF_cat_age')
plot_line(data=df_male, x_var='Age', y_var='SBP', indi='CRF_cat_age')
plot_line(data=df_male, x_var='Age', y_var='DBP', indi='CRF_cat_age')

# %%

### Age Category data size
df_male['Age'].value_counts()

# %%