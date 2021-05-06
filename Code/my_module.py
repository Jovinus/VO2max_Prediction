import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def adjusted_r2(model, x, y):
    yhat = model.predict(x)
    SS_Residual = sum((y-yhat)**2)
    SS_Total = sum((y-np.mean(y))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1)
    return adjusted_r_squared

def get_metric(model, X_train, y_train, X_test, y_test):
    print('Train set Adjusted R^2: {}'.format(adjusted_r2(model, X_train, y_train)))
    print("Train set Multiple Correlation: {}".format(np.power(model.score(X_train, y_train), 1/2)))
    print('Validation set Adjusted R^2: {}'.format(adjusted_r2(model, X_test, y_test)))
    print("Validation set Multiple Correlation: {}".format(np.power(model.score(X_test, y_test), 1/2)))
    print('Train set SEE: {}'.format(np.std(model.predict(X_train) - y_train)))
    print('Validation set SEE: {}'.format(np.std(model.predict(X_test) - y_test)))
    print('MSE Train set score: {}'.format(mean_squared_error(model.predict(X_train), y_train)))
    print('MSE Validation set score: {}'.format(mean_squared_error(model.predict(X_test), y_test)))
    return None

def plot_balnd_altman(model, x, y, file_nm):
    SAVEPATH = '/home/lkh256/Studio/VO2max_Prediction/Figure'
    
    y_hat = model.predict(x)

    f, ax = plt.subplots(1, figsize = (10, 10))
    sm.graphics.mean_diff_plot(y_hat, y, ax = ax)
    plt.savefig(os.path.join(SAVEPATH, file_nm), dpi = 300)
    
    return None

def make_model_get_metrics(X_train, y_train, X_test, y_test, column_mask):
    # linear_model = ElasticNet()
    # linear_model = Lasso()
    # linear_model = MLPRegressor(hidden_layer_sizes=(50, 50))
    linear_model = LinearRegression(n_jobs=-1)
    
    # scaler = StandardScaler()
    # X_train[column_mask] = scaler.fit_transform(X_train[column_mask])
    # X_test[column_mask] = scaler.transform(X_test[column_mask])
    
    linear_model.fit(X_train[column_mask], y_train)
    
    equation = "\nEquation : "
    
    for i, variable in enumerate(column_mask):
        equation = equation + str(linear_model.coef_[i]) + '*' + str(variable) + ' + '
        
    equation = equation + str(linear_model.intercept_)
    print(equation)
    
    get_metric(linear_model, X_train=X_train[column_mask], X_test=X_test[column_mask],
            y_train=y_train, y_test=y_test)
    
    plot_balnd_altman(model = linear_model, x=X_test[column_mask], y=y_test, file_nm="_".join(column_mask))
    
    return linear_model


def make_model_tertile(df_selected, df_surv, X_train, y_train, X_test, y_test, sex_specific=False):
    # %% Age. BMI, Rest_HR, MVPA
    if sex_specific == False:
        column_mask = ['AGE', 'sex', 'BMI', 'rest_HR', 'MVPA']
    else:
        column_mask = ['AGE', 'BMI', 'rest_HR', 'MVPA']

    model = make_model_get_metrics(X_train, y_train, X_test, y_test, column_mask)

    df_selected['ABRP_VO2max'] = model.predict(df_selected[column_mask])
    df_selected['ABRP_CRF'] = model.predict(df_selected[column_mask]) / 3.5

    df_surv['ABRP_VO2max'] = model.predict(df_surv[column_mask])
    df_surv['ABRP_CRF'] = model.predict(df_surv[column_mask]) / 3.5

    # %% Age. BMI, Rest_HR
    if sex_specific == False:
        column_mask = ['AGE', 'sex', 'BMI', 'rest_HR']
    else:
        column_mask = ['AGE', 'BMI', 'rest_HR']

    model = make_model_get_metrics(X_train, y_train, X_test, y_test, column_mask)

    df_selected['ABR_VO2max'] = model.predict(df_selected[column_mask])
    df_selected['ABR_CRF'] = model.predict(df_selected[column_mask]) / 3.5

    df_surv['ABR_VO2max'] = model.predict(df_surv[column_mask])
    df_surv['ABR_CRF'] = model.predict(df_surv[column_mask]) / 3.5

    # %% Age. BMI, MVPA
    if sex_specific == False:
        column_mask = ['AGE', 'sex', 'BMI', 'MVPA']
    else:
        column_mask = ['AGE', 'BMI', 'MVPA']

    model = make_model_get_metrics(X_train, y_train, X_test, y_test, column_mask)

    df_selected['ABP_VO2max'] = model.predict(df_selected[column_mask])
    df_selected['ABP_CRF'] = model.predict(df_selected[column_mask]) / 3.5

    df_surv['ABP_VO2max'] = model.predict(df_surv[column_mask])
    df_surv['ABP_CRF'] = model.predict(df_surv[column_mask]) / 3.5

    # %% Age. Percentage_fat, rest_HR, MVPA
    if sex_specific == False:
        column_mask = ['AGE', 'sex', 'percentage_fat', 'rest_HR', 'MVPA']
    else:
        column_mask = ['AGE', 'percentage_fat', 'rest_HR', 'MVPA']

    model = make_model_get_metrics(X_train, y_train, X_test, y_test, column_mask)

    df_selected['APRP_VO2max'] = model.predict(df_selected[column_mask])
    df_selected['APRP_CRF'] = model.predict(df_selected[column_mask]) / 3.5

    df_surv['APRP_VO2max'] = model.predict(df_surv[column_mask])
    df_surv['APRP_CRF'] = model.predict(df_surv[column_mask]) / 3.5

    # %% Age. Percentage_fat, rest_HR
    if sex_specific == False:
        column_mask = ['AGE', 'sex', 'percentage_fat', 'rest_HR']
    else:
        column_mask = ['AGE', 'percentage_fat', 'rest_HR']

    model = make_model_get_metrics(X_train, y_train, X_test, y_test, column_mask)

    df_selected['APR_VO2max'] = model.predict(df_selected[column_mask])
    df_selected['APR_CRF'] = model.predict(df_selected[column_mask]) / 3.5

    df_surv['APR_VO2max'] = model.predict(df_surv[column_mask])
    df_surv['APR_CRF'] = model.predict(df_surv[column_mask]) / 3.5

    # %% Age. Percentage_fat, MVPA
    if sex_specific == False:
        column_mask = ['AGE', 'sex', 'percentage_fat', 'MVPA']
    else:
        column_mask = ['AGE', 'percentage_fat', 'MVPA']

    model = make_model_get_metrics(X_train, y_train, X_test, y_test, column_mask)

    df_selected['APP_VO2max'] = model.predict(df_selected[column_mask])
    df_selected['APP_CRF'] = model.predict(df_selected[column_mask]) / 3.5

    df_surv['APP_VO2max'] = model.predict(df_surv[column_mask])
    df_surv['APP_CRF'] = model.predict(df_surv[column_mask]) / 3.5


    """
    -------------------------------- Devide estimates for survival analysis  -----------------------
    There is two types of model that estimate VO2max(CRF)
    - BMI
    - VO2max

    Adjusted with age, sex, rest_HR, MVPA
    ------------------------------------------------------------------------------------------------
    """
    

    ################################ Tertile ##################################

    #### Ref
    df_selected['CRF_tertile'] = pd.qcut(df_selected['CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_selected['CRF_tertile_nm'] = pd.qcut(df_selected['CRF'], q=3)

    df_surv['CRF_tertile'] = pd.qcut(df_surv['CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_surv['CRF_tertile_nm'] = pd.qcut(df_surv['CRF'], q=3)

    #### BMI - ABRP
    df_selected['ABRP_CRF_tertile'] = pd.qcut(df_selected['ABRP_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_selected['ABRP_CRF_tertile_nm'] = pd.qcut(df_selected['ABRP_CRF'], q=3)

    df_surv['ABRP_CRF_tertile'] = pd.qcut(df_surv['ABRP_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_surv['ABRP_CRF_tertile_nm'] = pd.qcut(df_surv['ABRP_CRF'], q=3)
    
    #### BMI - ABR
    df_selected['ABR_CRF_tertile'] = pd.qcut(df_selected['ABR_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_selected['ABR_CRF_tertile_nm'] = pd.qcut(df_selected['ABRP_CRF'], q=3)

    df_surv['ABR_CRF_tertile'] = pd.qcut(df_surv['ABR_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_surv['ABR_CRF_tertile_nm'] = pd.qcut(df_surv['ABR_CRF'], q=3)
    
    #### BMI - ABP
    df_selected['ABP_CRF_tertile'] = pd.qcut(df_selected['ABP_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_selected['ABP_CRF_tertile_nm'] = pd.qcut(df_selected['ABP_CRF'], q=3)

    df_surv['ABP_CRF_tertile'] = pd.qcut(df_surv['ABP_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_surv['ABP_CRF_tertile_nm'] = pd.qcut(df_surv['ABP_CRF'], q=3)

    #### Percentage Fat - APRP
    df_selected['APRP_CRF_tertile'] = pd.qcut(df_selected['APRP_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_selected['APRP_CRF_tertile_nm'] = pd.qcut(df_selected['APRP_CRF'], q=3)

    df_surv['APRP_CRF_tertile'] = pd.qcut(df_surv['APRP_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_surv['APRP_CRF_tertile_nm'] = pd.qcut(df_surv['APRP_CRF'], q=3)
    
    #### Percentage Fat - APR
    df_selected['APR_CRF_tertile'] = pd.qcut(df_selected['APR_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_selected['APR_CRF_tertile_nm'] = pd.qcut(df_selected['APR_CRF'], q=3)

    df_surv['APR_CRF_tertile'] = pd.qcut(df_surv['APR_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_surv['APR_CRF_tertile_nm'] = pd.qcut(df_surv['APR_CRF'], q=3)
    
    #### Percentage Fat - APP
    df_selected['APP_CRF_tertile'] = pd.qcut(df_selected['APP_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_selected['APP_CRF_tertile_nm'] = pd.qcut(df_selected['APP_CRF'], q=3)

    df_surv['APP_CRF_tertile'] = pd.qcut(df_surv['APP_CRF'], q=3, labels=['T1', 'T2', 'T3'])
    df_surv['APP_CRF_tertile_nm'] = pd.qcut(df_surv['APP_CRF'], q=3)

    ################################ Quantile #################################

    #### Ref
    df_selected['CRF_qualtile'] = pd.qcut(df_selected['CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_selected['CRF_qualtile_nm'] = pd.qcut(df_selected['CRF'], q=4)

    df_surv['CRF_qualtile'] = pd.qcut(df_surv['CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_surv['CRF_qualtile_nm'] = pd.qcut(df_surv['CRF'], q=4)

    #### BMI - ABRP
    df_selected['ABRP_CRF_qualtile'] = pd.qcut(df_selected['ABRP_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_selected['ABRP_CRF_qualtile_nm'] = pd.qcut(df_selected['ABRP_CRF'], q=4)

    df_surv['ABRP_CRF_qualtile'] = pd.qcut(df_surv['ABRP_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_surv['ABRP_CRF_qualtile_nm'] = pd.qcut(df_surv['ABRP_CRF'], q=4)
    
    #### BMI - ABR
    df_selected['ABR_CRF_qualtile'] = pd.qcut(df_selected['ABR_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_selected['ABR_CRF_qualtile_nm'] = pd.qcut(df_selected['ABR_CRF'], q=4)

    df_surv['ABR_CRF_qualtile'] = pd.qcut(df_surv['ABR_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_surv['ABR_CRF_qualtile_nm'] = pd.qcut(df_surv['ABR_CRF'], q=4)
    
    #### BMI - ABP
    df_selected['ABP_CRF_qualtile'] = pd.qcut(df_selected['ABP_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_selected['ABP_CRF_qualtile_nm'] = pd.qcut(df_selected['ABP_CRF'], q=4)

    df_surv['ABP_CRF_qualtile'] = pd.qcut(df_surv['ABP_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_surv['ABP_CRF_qualtile_nm'] = pd.qcut(df_surv['ABP_CRF'], q=4)

    #### Percentage Fat - APRP
    df_selected['APRP_CRF_qualtile'] = pd.qcut(df_selected['APRP_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_selected['APRP_CRF_qualtile_nm'] = pd.qcut(df_selected['APRP_CRF'], q=4)

    df_surv['APRP_CRF_qualtile'] = pd.qcut(df_surv['APRP_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_surv['APRP_CRF_qualtile_nm'] = pd.qcut(df_surv['APRP_CRF'], q=4)
    
    #### Percentage Fat - APR
    df_selected['APR_CRF_qualtile'] = pd.qcut(df_selected['APR_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_selected['APR_CRF_qualtile_nm'] = pd.qcut(df_selected['APR_CRF'], q=4)

    df_surv['APR_CRF_qualtile'] = pd.qcut(df_surv['APR_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_surv['APR_CRF_qualtile_nm'] = pd.qcut(df_surv['APR_CRF'], q=4)
    
    #### Percentage Fat - APP
    df_selected['APP_CRF_qualtile'] = pd.qcut(df_selected['APP_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_selected['APP_CRF_qualtile_nm'] = pd.qcut(df_selected['APP_CRF'], q=4)

    df_surv['APP_CRF_qualtile'] = pd.qcut(df_surv['APP_CRF'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_surv['APP_CRF_qualtile_nm'] = pd.qcut(df_surv['APP_CRF'], q=4)

    return df_selected, df_surv

