import torch
from datatable import fread
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset():
    DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Data"
    df_orig = fread(os.path.join(DATA_PATH, 'general_eq.csv'),
                    encoding='utf-8-sig', 
                    na_strings=['', 'NA']).to_pandas()
    
    df_orig['SM_DATE'] = df_orig['SM_DATE'].astype('datetime64')
    df_orig['visit_num'] = df_orig.groupby(['HPCID'])['SM_DATE'].apply(pd.Series.rank)
    df_orig = df_orig[(df_orig['visit_num'] == 1) & (df_orig['sex'] == 0)].reset_index(drop=True)
    
    #### basic data preprocessing
    categorical = ['sex', 'MVPA', 'Smoke', 'ALC']
    numeric = ['AGE', 'BMI_cal', 'percentage_fat', 'rest_HR', '체수분량']
    
    df_orig[categorical] = df_orig[categorical] * 1
    
    #### Define columns to analysis
    column_mask = numeric + categorical
    
    x = torch.from_numpy(df_orig[column_mask].values).float()
    y = torch.from_numpy(df_orig['VO2max'].values).float()
    
    return x, y

def split_data(x, y, train_ratio=0.8, device=torch.device('cpu')):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt
    
    # Shuffle dataset to split into train/valid set
    indices = torch.randperm(x.size(0))
    
    x = torch.index_select(
        x, 
        dim=0, 
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    
    y = torch.index_select(
        y, 
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    
    scaler = StandardScaler()
    scaler.fit(x[0].numpy())
    x = (torch.from_numpy(scaler.transform(x[0].numpy())).to(device), 
         torch.from_numpy(scaler.transform(x[1].numpy())).to(device))
    y = (torch.reshape(y[0], (-1, 1)).to(device), torch.reshape(y[1], (-1, 1)).to(device))
    
    return x, y


def get_hidden_sizes(input_size, output_size, n_layers, n_node_first_hidden):
    # step_size = int((input_size - output_size) / n_layers)
    
    step_size = int((n_node_first_hidden - output_size) / n_layers)
    
    hidden_sizes =[]
    current_size = n_node_first_hidden
    
    for i in range(n_layers - 1):
        if i == 0:
            hidden_sizes += [current_size]
        else:
            hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]
        
    return hidden_sizes

if __name__ == '__main__':
    x, y = load_dataset()
    x, y = split_data(x, y, train_ratio=0.8)
    print(x)
    print(y)
    
    print(get_hidden_sizes(100, 1, 3, 100))